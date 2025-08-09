# tts/openai_tts_engine.py

import time
from .base_engine import TTSEngineBase
from queue import Queue, Empty
import threading
import sounddevice as sd
import numpy as np
from collections import deque
from openai import OpenAI
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

# === Playback stability constants ===
SR = 24000
FRAME_MS = 10
BLOCKSIZE_SAMPLES = SR * FRAME_MS // 1000  # 240 samples = 10ms @ 24k


class OpenAITTSEngine(TTSEngineBase):
    def __init__(self, voice: str, stop_event: threading.Event, model: str = "tts-1", **kwargs):
        super().__init__(voice=voice, stop_event=stop_event)
        self.model = model
        self.client = OpenAI()
        self.sample_rate = SR  # 24kHz

        self.playback_thread = None
        self.stream_running = threading.Event()

        # Phase 1 options & debug
        self.jitter_frames = int(kwargs.get("jitter_frames", 1))  # 建議 0~12
        self.debug = bool(kwargs.get("debug", True))

        # Phase 2 feature flag：真正串流（iter_bytes→切幀→入列）
        self.stream_frames = bool(kwargs.get("stream_frames", True))
        # 串流預載門檻（幀數）：只在「每段語句開頭」檢查一次
        self.prebuffer_min_frames = int(kwargs.get("prebuffer_min_frames", 1))

        # 是否強制「單講者」（預設關閉，避免干擾測試）
        self.enforce_single_speaker = bool(
            kwargs.get("enforce_single_speaker", False))

        # 佇列大小（你原本的設定）
        max_q = 48 if self.stream_frames else 1
        self.pcm_data_queue = Queue(maxsize=max_q)

        # runtime counters / flags
        self._jitter_left = 0
        self._underflows = 0
        self._overflows = 0
        self._silence_fills = 0
        self._cb_calls = 0
        self._prebuffer_waits = 0  # 觀測預載等待次數

        # 只在「新一段」套 jitter & 預載
        self._apply_jitter_on_next_chunk = False
        self._prebuffer_active = False

        # 串流模式用來精準等待播放完成
        self._samples_enqueued = 0
        self._samples_played = 0
        self._producer_thread: Optional[threading.Thread] = None

        # Phase 3：輕量中斷旗標
        self._producer_cancel = threading.Event()  # 取消當前 producer
        self._drop_flag = threading.Event()        # callback 丟棄當前播放 buffer

        # 乾淨啟動（避免句尾與下一句無縫黏在一起）
        self.dry_start = bool(kwargs.get("dry_start", True))

        # === Duck/Unduck gain control ===
        self._gain_current: float = 1.0
        self._gain_target: float = 1.0
        self._gain_step: float = 0.0
        self._gain_transition_left: int = 0
        self._gain_total: int = 0
        self._gain_lock = threading.Lock()
        self._duck_attack_ms = int(kwargs.get("duck_attack_ms", 15))
        self._duck_release_ms = int(kwargs.get("duck_release_ms", 30))
        self._duck_db = float(kwargs.get("duck_db", -18.0))  # 預設 -18dB
        self._gain_eps = 1e-4  # 去抖用

        # === Pause/Resume (backlog) ===
        self._pause_lock = threading.Lock()
        self._paused: bool = False
        self._paused_since_ts: Optional[float] = None

        self._pause_drain_batch = int(kwargs.get(
            "pause_drain_batch", 8))  # 暫停時每回調最多吸幾幀

        self._backlog = deque()  # 存 numpy.int16 幀
        # 建議拉高到幾千幀（例如 6000 ≈ 60s）視 RAM 而定
        self._backlog_frames_limit = int(
            kwargs.get("backlog_frames_limit", 500))
        self._backlog_frames = 0
        self._backlog_drops = 0  # 因超出上限被丟掉的幀數

        # 這個實例的標識（方便 log 分辨）
        self.engine_tag = f"{voice}-{id(self) % 1000:03d}"

        # 單講者（process 級）控制：第一個 start_stream() 的實例成為 primary
        self._primary_id: Optional[int] = None
        self._primary_lock = threading.Lock()

    # ------------------ helpers ------------------ #
    def _drain_queue_nonblocking(self) -> int:
        """清空播放佇列（非阻塞），回傳丟掉的項數。"""
        dropped = 0
        try:
            while True:
                self.pcm_data_queue.get_nowait()
                dropped += 1
        except Empty:
            pass
        return dropped

    def _purge_for_new_utterance(self):
        """在新一段開始前，確保『絕對乾淨』：丟當前 buffer、清空佇列、對齊 enq/play。"""
        self._drop_flag.set()
        n = self._drain_queue_nonblocking()
        self._samples_enqueued = self._samples_played
        time.sleep(FRAME_MS / 1000.0)  # 等一個 callback 週期
        self._drop_flag.clear()
        if self.debug and n:
            print(f"[{self.engine_tag}] dry-start purge: dropped {n} queued frames")

    # === Gain transition control ===
    def _start_gain_transition(self, target_gain: float, ms: int):
        """開始從 current 向 target_gain 過渡，歷時 ms 毫秒（具去抖）。"""
        target_gain = float(max(0.0, min(1.0, target_gain)))
        total = max(1, int(self.sample_rate * (ms / 1000.0)))

        with self._gain_lock:
            if abs(target_gain - self._gain_target) < self._gain_eps and self._gain_transition_left == 0:
                if self.debug:
                    print(
                        f"[{self.engine_tag}] GAIN noop (target unchanged: {target_gain:.3f})")
                return
            self._gain_target = target_gain
            self._gain_total = total
            self._gain_transition_left = total
            self._gain_step = (self._gain_target -
                               self._gain_current) / float(total)

        if self.debug:
            print(f"[{self.engine_tag}] GAIN transition: current={self._gain_current:.3f} -> "
                  f"target={self._gain_target:.3f} in {ms}ms ({total} samples)")

    def duck(self, db: float = None):
        if db is None:
            db = self._duck_db
        target = 10.0 ** (float(db) / 20.0)
        self._start_gain_transition(
            target_gain=target, ms=self._duck_attack_ms)

    def unduck(self):
        self._start_gain_transition(target_gain=1.0, ms=self._duck_release_ms)

    # === backlog primitives ===
    def _backlog_push(self, frame: np.ndarray):
        with self._pause_lock:
            if self._backlog_frames >= self._backlog_frames_limit:
                try:
                    self._backlog.popleft()
                    self._backlog_drops += 1
                    self._backlog_frames -= 1
                except IndexError:
                    pass
            self._backlog.append(frame)
            self._backlog_frames += 1

    def _backlog_pop(self) -> Optional[np.ndarray]:
        with self._pause_lock:
            try:
                fr = self._backlog.popleft()
                self._backlog_frames -= 1
                return fr
            except IndexError:
                return None

    def _backlog_clear(self):
        with self._pause_lock:
            self._backlog.clear()
            self._backlog_frames = 0

    def _backlog_len(self) -> int:
        with self._pause_lock:
            return self._backlog_frames

    # === Public pause / resume ===
    def pause_output(self, flush: bool = False):
        """暫停『輸出』：仍消費 queue 並寫入 backlog。絕不自動恢復。"""
        with self._pause_lock:
            if not self._paused:
                self._paused = True
                self._paused_since_ts = time.time()
        if flush:
            # 清空 backlog
            self._backlog_clear()
            # 清空播放 queue（避免殘留片段回放）
            self._drain_queue_nonblocking()
            # 對齊計數，避免等待條件被卡
            self._samples_enqueued = self._samples_played

        if self.debug:
            print(f"[{self.engine_tag}] OUTPUT PAUSED (flush={flush}).")

    def resume_output(self, flush: bool = False):
        """恢復輸出；flush=True 會清空 backlog（用於確定要中斷時）。"""
        if flush:
            self._backlog_clear()
            self._samples_enqueued = self._samples_played

        with self._pause_lock:
            self._paused = False
            self._paused_since_ts = None
        if self.debug:
            print(f"[{self.engine_tag}] OUTPUT RESUMED (flush={flush}).")

    # ------------------ Playback loop ------------------ #
    def _playback_loop(self):
        current_pcm_data = None
        current_position = 0

        def _apply_gain_inplace_int16(outbuf: np.ndarray):
            with self._gain_lock:
                gain_cur = float(self._gain_current)
                gain_tgt = float(self._gain_target)
                step = float(self._gain_step)
                trans_left = int(self._gain_transition_left)

            frames = outbuf.shape[0]
            if trans_left > 0:
                n = min(frames, trans_left)
                if n > 0:
                    ramp = (gain_cur + step * np.arange(n, dtype=np.float32))
                    tmp = outbuf[:n, 0].astype(np.float32)
                    tmp *= ramp
                    np.clip(tmp, -32768, 32767, out=tmp)
                    outbuf[:n, 0] = tmp.astype(np.int16)
                    gain_cur += step * n
                    trans_left -= n

                if frames > n:
                    g = gain_cur if trans_left > 0 else gain_tgt
                    if g != 1.0:
                        tmp2 = outbuf[n:, 0].astype(np.float32)
                        tmp2 *= g
                        np.clip(tmp2, -32768, 32767, out=tmp2)
                        outbuf[n:, 0] = tmp2.astype(np.int16)

                with self._gain_lock:
                    self._gain_current = gain_cur if trans_left > 0 else gain_tgt
                    self._gain_transition_left = trans_left
                    if trans_left == 0:
                        self._gain_step = 0.0
            else:
                g = gain_cur
                if g != 1.0:
                    tmp = outbuf[:, 0].astype(np.float32)
                    tmp *= g
                    np.clip(tmp, -32768, 32767, out=tmp)
                    outbuf[:, 0] = tmp.astype(np.int16)

        def callback(outdata, frames, time_info, status):
            nonlocal current_pcm_data, current_position
            self._cb_calls += 1

            if status.output_underflow:
                self._underflows += 1
            if status.output_overflow:
                self._overflows += 1

            # 要求丟棄當前 utterance
            if self._drop_flag.is_set():
                current_pcm_data = None
                current_position = 0
                outdata.fill(0)
                self._drain_queue_nonblocking()
                return

            # 句首 jitter
            if self._jitter_left > 0:
                outdata.fill(0)
                self._jitter_left -= 1
                _apply_gain_inplace_int16(outdata)
                return

            # 段首預載
            if (
                self.stream_frames
                and self._prebuffer_active
                and current_pcm_data is None
                and self._jitter_left == 0
            ):
                try:
                    qsz = self.pcm_data_queue.qsize()
                except Exception:
                    qsz = 0
                if qsz < self.prebuffer_min_frames:
                    outdata.fill(0)
                    self._silence_fills += 1
                    self._prebuffer_waits += 1
                    if self.debug and (self._cb_calls % 100 == 0):
                        print(
                            f"[{self.engine_tag}] prebuffer waiting: qsize={qsz}/{self.prebuffer_min_frames}")
                    _apply_gain_inplace_int16(outdata)
                    return
                else:
                    self._prebuffer_active = False
                    if self.debug:
                        print(
                            f"[{self.engine_tag}] prebuffer satisfied (qsize={qsz}), start playback")

            # === 暫停模式：不輸出；吸入 backlog ===
            with self._pause_lock:
                paused_now = self._paused

            if paused_now:
                # 把當前未播完的 buffer 尾端放進 backlog
                if current_pcm_data is not None:
                    remaining = len(current_pcm_data) - current_position
                    if remaining > 0:
                        self._backlog_push(
                            current_pcm_data[current_position:].copy())
                    current_pcm_data = None
                    current_position = 0

                # 吸收幾幀 queue → backlog，避免 producer 背壓
                for _ in range(self._pause_drain_batch):
                    try:
                        item = self.pcm_data_queue.get_nowait()
                        if isinstance(item, np.ndarray):
                            self._backlog_push(item)
                        elif item is None:
                            raise sd.CallbackStop
                    except Empty:
                        break

                outdata.fill(0)
                _apply_gain_inplace_int16(outdata)
                return

            # 若有 backlog，優先播 backlog
            if current_pcm_data is None:
                b = self._backlog_pop()
                if b is not None:
                    current_pcm_data = b
                    current_position = 0

            # 取 live queue
            if current_pcm_data is None:
                try:
                    item = self.pcm_data_queue.get_nowait()
                    if isinstance(item, np.ndarray):
                        current_pcm_data = item
                        current_position = 0

                        if self._apply_jitter_on_next_chunk:
                            self._apply_jitter_on_next_chunk = False
                            self._jitter_left = int(self.jitter_frames)
                            if self._jitter_left > 0:
                                outdata.fill(0)
                                self._jitter_left -= 1
                                _apply_gain_inplace_int16(outdata)
                                return

                        if self.debug and (self._cb_calls % 50 == 1):
                            print(
                                f"[{self.engine_tag}] Dequeued PCM len={len(current_pcm_data)}")

                    elif item is None:
                        raise sd.CallbackStop
                except Empty:
                    outdata.fill(0)
                    self._silence_fills += 1
                    _apply_gain_inplace_int16(outdata)
                    return

            # 固定幀長：frames == 240
            chunk_size = frames
            remaining = len(current_pcm_data) - current_position

            if remaining >= chunk_size:
                outdata[:, 0] = current_pcm_data[current_position: current_position + chunk_size]
                current_position += chunk_size

                if current_position >= len(current_pcm_data):
                    current_pcm_data = None
                    current_position = 0

                if self.stream_frames:
                    self._samples_played += chunk_size

            else:
                if remaining > 0:
                    outdata[:remaining, 0] = current_pcm_data[current_position:]
                    outdata[remaining:, 0] = 0
                    if self.stream_frames:
                        self._samples_played += remaining
                else:
                    outdata.fill(0)
                    self._silence_fills += 1
                current_pcm_data = None
                current_position = 0

            # 套用增益
            _apply_gain_inplace_int16(outdata)

            # 偵錯輸出（節流）
            if self.debug and (self._cb_calls % 200 == 0):
                qsz = getattr(self.pcm_data_queue, "qsize", lambda: -1)()
                print(
                    f"[{self.engine_tag}] cb#{self._cb_calls} frames={frames} "
                    f"qsize={qsz} backlog={self._backlog_len()} drops={self._backlog_drops} "
                    f"under={self._underflows} over={self._overflows} "
                    f"silence={self._silence_fills} prebuf_waits={self._prebuffer_waits} "
                    + (f"enq={self._samples_enqueued} play={self._samples_played}" if self.stream_frames else "")
                )

        print(
            f"[{self.engine_tag}] Starting persistent audio stream... "
            f"(sr={self.sample_rate}, blocksize={BLOCKSIZE_SAMPLES}, "
            f"jitter_frames={self.jitter_frames}, stream_frames={self.stream_frames}, "
            f"prebuffer_min_frames={self.prebuffer_min_frames})"
        )
        try:
            with sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='int16',
                blocksize=BLOCKSIZE_SAMPLES,
                latency='low',
                callback=callback
            ):
                self.stream_running.set()
                self.stop_event.wait()
            print(f"[{self.engine_tag}] Persistent audio stream stopped.")
        except Exception as e:
            print(f"[{self.engine_tag}] Audio stream error: {e}")
        finally:
            self.stream_running.clear()
            if self.debug:
                print(
                    f"[{self.engine_tag}] stats: cb_calls={self._cb_calls}, "
                    f"underflows={self._underflows}, overflows={self._overflows}, "
                    f"silence_fills={self._silence_fills}, prebuffer_waits={self._prebuffer_waits}, "
                    f"backlog={self._backlog_len()} drops={self._backlog_drops}"
                )

    # ------------------ Lifecycle ------------------ #
    def start_stream(self):
        if self.enforce_single_speaker:
            with self._primary_lock:
                if self._primary_id is None:
                    self._primary_id = id(self)
                    if self.debug:
                        print(f"[{self.engine_tag}] (primary) acquired")

        if self.playback_thread and self.playback_thread.is_alive():
            return
        print(f"[{self.engine_tag}] Start stream command received.")
        self.stop_event.clear()
        self.playback_thread = threading.Thread(
            target=self._playback_loop, daemon=True)
        self.playback_thread.start()
        self.stream_running.wait(timeout=5.0)

    def stop_stream(self):
        if not self.playback_thread or not self.playback_thread.is_alive():
            return
        print(f"[{self.engine_tag}] Stop stream command received.")
        self.stop_event.set()
        try:
            self.pcm_data_queue.put_nowait(None)  # 讓 callback 結束
        except Exception:
            pass
        self.playback_thread.join(timeout=2.0)
        self.playback_thread = None

        if self.enforce_single_speaker and self._primary_id == id(self):
            with self._primary_lock:
                self._primary_id = None
                if self.debug:
                    print(f"[{self.engine_tag}] (primary) released")

    # ------------------ Non-stream path ------------------ #
    def _fetch_and_decode_audio(self, text: str) -> Optional[np.ndarray]:
        print(f"[{self.engine_tag}] Fetching PCM for text: '{text[:20]}...'")
        response = self.client.audio.speech.create(
            model=self.model, voice=self.voice, response_format="pcm", input=text
        )
        pcm_data = np.frombuffer(response.content, dtype=np.int16).copy()
        if self.debug:
            print(f"[{self.engine_tag}] _fetch_and_decode_audio: len={len(pcm_data)}, "
                  f"duration≈{len(pcm_data)/self.sample_rate:.2f}s")
        return pcm_data

    # ------------------ Stream path ------------------ #
    def _produce_stream_frames(self, text: str):
        if self.debug:
            print(f"[{self.engine_tag}] Streaming TTS start for: '{text[:20]}...'")

        buf = bytearray()
        BYTES_PER_SAMPLE = 2
        BYTES_PER_FRAME = BLOCKSIZE_SAMPLES * BYTES_PER_SAMPLE

        try:
            with self.client.audio.speech.with_streaming_response.create(
                model=self.model,
                voice=self.voice,
                input=text,
                response_format="pcm",
            ) as resp:
                for chunk in resp.iter_bytes():
                    if self.stop_event.is_set() or self._producer_cancel.is_set():
                        if self.debug:
                            print(
                                f"[{self.engine_tag}] Streaming stopped (cancel/stop_event).")
                        return

                    buf.extend(chunk)

                    while len(buf) >= BYTES_PER_FRAME:
                        if self._producer_cancel.is_set():
                            if self.debug:
                                print(
                                    f"[{self.engine_tag}] Producer canceled while slicing frames.")
                            return
                        block = bytes(buf[:BYTES_PER_FRAME])
                        del buf[:BYTES_PER_FRAME]
                        frame = np.frombuffer(block, dtype='<i2').copy()
                        self.pcm_data_queue.put(frame)
                        self._samples_enqueued += len(frame)

                # flush 最後殘留
                if not self._producer_cancel.is_set() and len(buf) > 0:
                    pad_len = (BYTES_PER_FRAME - len(buf)) % BYTES_PER_FRAME
                    if pad_len:
                        buf.extend(b'\x00' * pad_len)
                    block = bytes(buf[:BYTES_PER_FRAME])
                    frame = np.frombuffer(block, dtype='<i2').copy()
                    self.pcm_data_queue.put(frame)
                    self._samples_enqueued += len(frame)

        except Exception as e:
            print(f"[{self.engine_tag}] Stream producer error: {e}")
        finally:
            if self.debug:
                print(f"[{self.engine_tag}] Streaming producer finished. "
                      f"enqueued_samples={self._samples_enqueued}")

    # ------------------ Public API ------------------ #
    def say(self, text: str):
        if not self.stream_running.is_set():
            print(f"[{self.engine_tag}] ERROR: Stream is not running.")
            return

        if self.enforce_single_speaker and self._primary_id not in (None, id(self)):
            if self.debug:
                print(f"[{self.engine_tag}] (non-primary) skip say()")
            return

        print(f"[{self.engine_tag}] Servicing TTS request for: '{text[:20]}...'")

        # 安全網：新句開播前保證恢復到正常音量（避免上一輪 duck 遺留）
        self.unduck()

        try:
            if not self.stream_frames:
                if self.debug:
                    print("[DEBUG] say: Fetching full PCM (non-stream mode)...")

                if self.dry_start:
                    self._purge_for_new_utterance()

                pcm_data = self._fetch_and_decode_audio(text)
                if pcm_data is not None:
                    self._apply_jitter_on_next_chunk = True
                    self._prebuffer_active = False
                    self.pcm_data_queue.put(pcm_data)

                    expected_duration = len(pcm_data) / self.sample_rate
                    wait_time = expected_duration + 0.5
                    if self.debug:
                        print(
                            f"[{self.engine_tag}] Audio queued. Waiting for playback (≈{wait_time:.2f}s)...")

                    start_time = time.time()
                    while time.time() - start_time < wait_time:
                        if self.stop_event.is_set():
                            print(
                                f"[{self.engine_tag}] Playback interrupted by stop event.")
                            break
                        time.sleep(0.05)

                    if self.debug:
                        print("[DEBUG] say: Wait loop finished (non-stream).")

            else:
                if self.dry_start:
                    self._purge_for_new_utterance()

                self._samples_enqueued = 0
                self._samples_played = 0
                self._apply_jitter_on_next_chunk = True
                self._prebuffer_active = True

                self._producer_cancel.clear()
                self._drop_flag.clear()

                if self._producer_thread and self._producer_thread.is_alive():
                    if self.debug:
                        print(
                            f"[{self.engine_tag}] Waiting previous producer to finish...")
                    self._producer_thread.join(timeout=5.0)

                self._producer_thread = threading.Thread(
                    target=self._produce_stream_frames, args=(
                        text,), daemon=True
                )
                self._producer_thread.start()

                if self.debug:
                    print(
                        f"[{self.engine_tag}] Stream producer started; waiting for playback...")

                # 等「producer 結束 + 播放完 queue + 播放完 backlog」
                while True:
                    if self.stop_event.is_set():
                        print(
                            f"[{self.engine_tag}] Interrupted by stop_event during stream.")
                        break

                    if self._producer_cancel.is_set():
                        break

                    producer_alive = self._producer_thread.is_alive() if self._producer_thread else False
                    try:
                        qsz = self.pcm_data_queue.qsize()
                    except Exception:
                        qsz = 0

                    backlog_empty = (self._backlog_len() == 0)
                    all_played = (not producer_alive) and (qsz == 0) and backlog_empty and \
                                 (self._samples_played >= self._samples_enqueued)

                    if all_played:
                        break

                    time.sleep(0.01)

                if self.debug:
                    print(f"[{self.engine_tag}] Stream say() finished: "
                          f"enq={self._samples_enqueued}, played={self._samples_played}, "
                          f"backlog={self._backlog_len()} drops={self._backlog_drops}")

        except Exception as e:
            print(f"[{self.engine_tag}] Error in say method: {e}")

        print(f"[{self.engine_tag}] SAY method finished for this text.")

    # ------------------ Phase 3: 輕量中斷 ------------------ #
    def stop_current_utterance(self):
        """中止『當前這句』的輸出：不關 stream。"""
        if self.debug:
            print(
                f"[{self.engine_tag}] stop_current_utterance(): cancel producer + clear queue + drop current buffer")

        self._producer_cancel.set()
        self._drain_queue_nonblocking()
        self._drop_flag.set()
        self._backlog_clear()  # 也要清掉 backlog
        self._samples_enqueued = self._samples_played

        if self._producer_thread and self._producer_thread.is_alive():
            self._producer_thread.join(timeout=0.3)

        time.sleep(FRAME_MS / 1000.0)
        self._drop_flag.clear()
        self._producer_cancel.clear()

        # 若當下在 paused 狀態，也一起解除
        with self._pause_lock:
            self._paused = False
            self._paused_since_ts = None

    def stop(self):
        print(f"[{self.engine_tag}] STOP command received, clearing queue.")
        self._drain_queue_nonblocking()
        self._backlog_clear()
        self.stop_event.set()
