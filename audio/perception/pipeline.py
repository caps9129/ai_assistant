# audio/pipeline.py

import threading
import queue
import numpy as np
import json
import time
from pathlib import Path
from collections import deque
from typing import Callable, Optional

from vad.webrtc import WebRTCVAD
from transcribers.router import AsrRouter


class PerceptionPipeline:
    def __init__(
        self,
        asr_router: AsrRouter,
        on_result_callback: Callable,
        *,
        # VAD 事件
        on_speech_onset: Optional[Callable[[], None]
                                  ] = None,      # 建議綁 router.duck
        # 建議綁 router.pause_output(flush=False)
        on_speech_commit: Optional[Callable[[], None]] = None,
        on_speech_cancel: Optional[Callable[[], None]
                                   ] = None,     # 建議綁 router.unduck

        # ASR 決策分流
        # 綁 router.stop_current_utterance
        on_stop_current_utterance: Optional[Callable[[], None]] = None,
        # 綁 router.resume_output(flush=False)
        on_resume_continue: Optional[Callable[[], None]] = None,
        # 綁 router.resume_output(flush=True)
        on_resume_flush: Optional[Callable[[], None]] = None,

        # 靈敏度參數
        commit_min_ms: int = 300,            # 建議先拉高
        onset_duck_delay_ms: int = 120,      # 連續語音滿這段才 duck
        window_frames: int = 8,              # 平滑視窗 M
        window_min_speech: int = 6,          # 至少 N 個 True 才算語音
        vad_frame_ms: int = 30,              # 若要更細可改 10ms（可選）
    ):
        # 讀取設定
        config_path = Path(__file__).parent.parent / "config.json"
        cfg = json.load(config_path.open("r"))
        self.sample_rate = cfg["sample_rate"]
        min_silence_duration_ms = cfg["min_silence_duration_ms"]

        self.asr_router = asr_router
        self.on_result_callback = on_result_callback

        # callbacks
        self._cb_onset = on_speech_onset
        self._cb_commit = on_speech_commit
        self._cb_cancel = on_speech_cancel
        self._cb_stop_curr = on_stop_current_utterance
        self._cb_resume_continue = on_resume_continue
        self._cb_resume_flush = on_resume_flush

        # VAD 設定
        self.vad_engine = WebRTCVAD(
            sample_rate=self.sample_rate, frame_duration_ms=int(vad_frame_ms), aggressiveness=3
        )
        self.frame_ms = self.vad_engine.frame_duration_ms

        # 句尾靜音門檻
        self.silence_frames_threshold = int(
            min_silence_duration_ms / self.frame_ms)

        # 門檻：commit / onset duck 延遲（換算成 frame 數）
        self.commit_min_ms = int(commit_min_ms)
        self.commit_min_frames = max(
            1, (self.commit_min_ms + self.frame_ms - 1) // self.frame_ms)

        self.onset_duck_delay_ms = int(onset_duck_delay_ms)
        self.onset_duck_delay_frames = max(
            1, (self.onset_duck_delay_ms + self.frame_ms - 1) // self.frame_ms)

        # N-out-of-M 平滑視窗
        self.window_frames = max(1, int(window_frames))
        self.window_min_speech = max(1, int(window_min_speech))
        self._speech_window = deque(maxlen=self.window_frames)

        # 內部佇列 / 執行緒
        self.audio_queue = queue.Queue()
        self.processing_thread = threading.Thread(
            target=self._process_loop, daemon=True)
        self.asr_queue = queue.Queue()
        self.asr_worker_thread = threading.Thread(
            target=self._asr_loop, daemon=True)

        self.is_running = False
        self.status_lock = threading.Lock()
        self._is_processing = False
        self._is_paused = True

    @property
    def is_processing(self):
        with self.status_lock:
            return self._is_processing

    def _set_processing_status(self, status: bool):
        with self.status_lock:
            self._is_processing = status

    def pause(self):
        with self.status_lock:
            if not self._is_paused:
                print("[Perception] Pausing pipeline.")
                self._is_paused = True

    def resume(self):
        with self.status_lock:
            if self._is_paused:
                print("[Perception] Resuming pipeline.")
                self._is_paused = False

    # ----------------------- ASR Worker -----------------------
    def _asr_loop(self):
        while self.is_running:
            audio_bytes, vad_end_time = self.asr_queue.get()
            if audio_bytes is None:
                continue

            print("[Perception] Running ASR fine-screening...")
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            result = self.asr_router.transcribe(audio_int16.tobytes())

            asr_end_time = time.time()
            latency = (asr_end_time - vad_end_time) * 1000
            print(
                f"LATENCY_METRIC: Perception (VAD end -> ASR result) = {latency:.2f} ms")

            text = (result or {}).get("text", "").strip()

            if text:
                # ===== ASR 有效：真正插話 =====
                print(
                    "[Perception] ASR valid -> stop_current_utterance + resume(flush=True) + unduck.")
                if self._cb_stop_curr:
                    try:
                        self._cb_stop_curr()
                    except Exception as e:
                        print(
                            f"[Perception] on_stop_current_utterance error: {e}")

                if self._cb_resume_flush:
                    try:
                        self._cb_resume_flush()   # router.resume_output(flush=True)
                    except Exception as e:
                        print(f"[Perception] on_resume_flush error: {e}")

                if self._cb_cancel:
                    try:
                        self._cb_cancel()         # router.unduck()
                    except Exception as e:
                        print(f"[Perception] on_speech_cancel error: {e}")

                print(f"[Perception] ASR result is valid: '{text}'")
                self.on_result_callback(result)

            else:
                # ===== ASR 無效：噪音/咳嗽 =====
                print(
                    "[Perception] ASR empty -> resume_output(flush=False) + unduck.")
                if self._cb_resume_continue:
                    try:
                        self._cb_resume_continue()  # router.resume_output(flush=False)
                    except Exception as e:
                        print(f"[Perception] on_resume_continue error: {e}")
                if self._cb_cancel:
                    try:
                        self._cb_cancel()           # router.unduck()
                    except Exception as e:
                        print(f"[Perception] on_speech_cancel error: {e}")
                print("[Perception] ASR result is empty, ignored as noise.")

            self._set_processing_status(False)

    # --------------------- VAD / FSM loop ---------------------
    def _process_loop(self):
        """
        VAD + onset/commit/cancel + 平滑投票：
        - onset（延遲）：連續語音滿 onset_duck_delay_ms 才觸發 on_speech_onset（duck）
        - commit：連續語音滿 commit_min_ms 觸發 on_speech_commit（pause_output）
        - 未達門檻回靜音：on_speech_cancel（unduck，不送 ASR）
        """
        in_speech = False
        committed = False
        speech_run_frames = 0
        onset_acc_frames = 0
        onset_duck_called = False

        audio_buffer = bytearray()
        silent_frames_count = 0

        while self.is_running:
            try:
                with self.status_lock:
                    if self._is_paused:
                        time.sleep(0.1)
                        # reset
                        in_speech = False
                        committed = False
                        speech_run_frames = 0
                        onset_acc_frames = 0
                        onset_duck_called = False
                        audio_buffer.clear()
                        silent_frames_count = 0
                        self._speech_window.clear()
                        continue

                audio_chunk = self.audio_queue.get(timeout=0.2)
                frames = self.vad_engine.process_chunk(audio_chunk)

                for raw_is_speech, frame_bytes in frames:
                    # ---- N-out-of-M 平滑 ----
                    self._speech_window.append(1 if raw_is_speech else 0)
                    smooth_speech = sum(
                        self._speech_window) >= self.window_min_speech

                    if smooth_speech:
                        # --- 有聲幀（平滑後） ---
                        if not in_speech:
                            print(
                                f"[Perception] VAD onset (≥{self.window_min_speech}/{self.window_frames}).")
                            in_speech = True
                            committed = False
                            speech_run_frames = 0
                            onset_acc_frames = 0
                            onset_duck_called = False
                            self._set_processing_status(True)

                        audio_buffer.extend(frame_bytes)
                        speech_run_frames += 1
                        onset_acc_frames += 1
                        silent_frames_count = 0

                        # 延遲 duck：避免短促爆音
                        if (not onset_duck_called) and (onset_acc_frames >= self.onset_duck_delay_frames):
                            onset_duck_called = True
                            if self._cb_onset:
                                try:
                                    self._cb_onset()
                                    print(
                                        f"[Perception] Duck after {onset_acc_frames * self.frame_ms} ms sustained speech.")
                                except Exception as e:
                                    print(
                                        f"[Perception] on_speech_onset error: {e}")

                        # 達到 commit 門檻
                        if (not committed) and (speech_run_frames >= self.commit_min_frames):
                            committed = True
                            print(
                                f"[Perception] Speech committed at {speech_run_frames * self.frame_ms} ms.")
                            if self._cb_commit:
                                try:
                                    self._cb_commit()
                                except Exception as e:
                                    print(
                                        f"[Perception] on_speech_commit error: {e}")

                    else:
                        # --- 無聲（平滑後） ---
                        if in_speech:
                            silent_frames_count += 1

                            # 未 commit 就回靜音 → cancel（不送 ASR）
                            if not committed and speech_run_frames > 0 and silent_frames_count == 1:
                                print(
                                    "[Perception] Speech canceled before commit.")
                                if self._cb_cancel:
                                    try:
                                        self._cb_cancel()
                                    except Exception as e:
                                        print(
                                            f"[Perception] on_speech_cancel error: {e}")
                                self._set_processing_status(False)
                                # reset
                                in_speech = False
                                committed = False
                                speech_run_frames = 0
                                onset_acc_frames = 0
                                onset_duck_called = False
                                audio_buffer.clear()
                                silent_frames_count = 0
                                continue

                            # 句尾靜音 → 送 ASR
                            if silent_frames_count > self.silence_frames_threshold:
                                vad_end_time = time.time()
                                print(
                                    f"[Perception] VAD finalized a segment due to silence at {vad_end_time:.2f}, sending to ASR worker.")
                                self.asr_queue.put(
                                    (audio_buffer.copy(), vad_end_time))
                                # reset
                                in_speech = False
                                committed = False
                                speech_run_frames = 0
                                onset_acc_frames = 0
                                onset_duck_called = False
                                audio_buffer.clear()
                                silent_frames_count = 0

            except queue.Empty:
                if in_speech:
                    vad_end_time = time.time()
                    print(
                        f"[Perception] Finalizing segment due to inactivity at {vad_end_time:.2f}, sending to ASR worker.")
                    self.asr_queue.put((audio_buffer.copy(), vad_end_time))
                    in_speech = False
                    committed = False
                    speech_run_frames = 0
                    onset_acc_frames = 0
                    onset_duck_called = False
                    audio_buffer.clear()
                    silent_frames_count = 0
                continue

    # ----------------------- Public APIs -----------------------
    def process_audio(self, audio_chunk: np.ndarray):
        if self.is_running:
            self.audio_queue.put(audio_chunk)

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.processing_thread.start()
            self.asr_worker_thread.start()
            print(
                f"✅ Perception Pipeline started. frame_ms={self.frame_ms}, "
                f"window={self.window_min_speech}/{self.window_frames}, "
                f"onset_delay_ms={self.onset_duck_delay_ms}, "
                f"commit_min_ms={self.commit_min_ms}({self.commit_min_frames} frames), "
                f"silence_ms≈{self.silence_frames_threshold * self.frame_ms}"
            )

    def stop(self):
        if self.is_running:
            self.is_running = False
            self.asr_queue.put((None, None))
            self.processing_thread.join(timeout=1.0)
            self.asr_worker_thread.join(timeout=1.0)
            print("🛑 Perception Pipeline stopped.")
