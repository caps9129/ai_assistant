# utils/sound_player.py

import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import time


class SoundPlayer:
    def __init__(self, sound_file_path, *, volume: float = 1.0,
                 block_ms: int = 10, loop_gap_ms: int = 0, latency='low'):
        """
        volume: 0.0~1.0 音量倍率
        block_ms: 回調幀長，預設 10ms → 停播反應更快
        loop_gap_ms: 每次循環之間插入的靜音（預設 0 無縫）
        latency: 傳給 sounddevice 的 latency 參數（'low' 反應較快）
        """
        self.sound_file_path = sound_file_path
        self.volume = float(volume)
        self.latency = latency

        try:
            # 讀檔並強制轉單聲道 float32，確保 (N,1) 形狀且連續
            data, fs = sf.read(self.sound_file_path,
                               dtype='float32', always_2d=True)
            if data.shape[1] > 1:
                data = data.mean(axis=1, keepdims=True)  # 轉 mono
            self.data = np.ascontiguousarray(data)      # (N, 1) float32
            self.fs = int(fs)
            print(
                f"[SoundPlayer] Loaded '{sound_file_path}' @ {self.fs} Hz, frames={self.data.shape[0]}")
        except Exception as e:
            print(
                f"[SoundPlayer] Error loading file {self.sound_file_path}: {e}")
            self.data, self.fs = None, 0

        # 播放控制
        self.stop_event = threading.Event()
        self._thread = None
        self._stream = None
        self._lock = threading.Lock()

        # 參數化回調設定
        self.blocksize = max(
            1, int(self.fs * block_ms / 1000)) if self.fs else 0
        self.loop_gap_frames = int(
            self.fs * loop_gap_ms / 1000) if self.fs else 0

        # 淡出控制
        self._fade_req = False
        self._fade_samples_left = 0

    def _loop_runner(self):
        if self.data is None or self.fs == 0:
            return

        pos = 0  # 目前播放到第幾個 frame
        gap_left = 0  # 檔與檔之間要插入的靜音幀

        def callback(outdata, frames, time_info, status):
            nonlocal pos, gap_left
            if status:
                print(f"[SoundPlayer] status: {status}")

            out = outdata  # shape (frames, 1)
            out.fill(0)

            i = 0
            while i < frames:
                if self.stop_event.is_set():
                    # 立刻停止：清空後中止
                    out[i:, :] = 0
                    raise sd.CallbackAbort

                # 淡出：將輸出乘上線性權重，直到倒數完
                if self._fade_req and self._fade_samples_left > 0:
                    # 先正常取樣，再乘上淡出權重
                    # 如果淡出剩餘小於此次 frames，就只淡出一部分
                    apply_fade = min(frames - i, self._fade_samples_left)
                else:
                    apply_fade = 0  # 不需要特殊處理（後面直接寫）

                # 插入循環間隔靜音
                if gap_left > 0:
                    n = min(frames - i, gap_left)
                    # out[i:i+n] 已是 0
                    gap_left -= n
                    i += n
                    continue

                # 從音檔拷貝
                remaining = self.data.shape[0] - pos
                n = min(frames - i, remaining)
                if n > 0:
                    out[i:i+n, 0] = self.data[pos:pos+n, 0] * self.volume
                    pos += n
                    # 需要淡出時，對這段尾巴乘權重
                    if apply_fade:
                        # 計算這段中需要淡出的子段
                        fade_n = min(n, apply_fade)
                        # 生成線性淡出權重（從 1 到 0）
                        # 若連續回調淡出，維持遞減
                        start = self._fade_samples_left - fade_n
                        if start < 0:
                            start = 0
                            fade_n = self._fade_samples_left
                        w = np.linspace(1.0 * start / self._fade_samples_left,
                                        0.0, fade_n, endpoint=False, dtype=np.float32)
                        out[i + (n - fade_n): i + n, 0] *= w
                        self._fade_samples_left -= fade_n
                        if self._fade_samples_left <= 0:
                            # 淡出完成 → 要求立即終止
                            self.stop_event.set()
                    i += n

                # 到檔尾，環回並插入靜音（若設定）
                if pos >= self.data.shape[0]:
                    pos = 0
                    gap_left = self.loop_gap_frames

            # 正常結束：什麼都不做，讓迴圈繼續

        try:
            with self._lock:
                self._stream = sd.OutputStream(
                    samplerate=self.fs,
                    channels=1,
                    dtype='float32',
                    blocksize=self.blocksize or None,
                    latency=self.latency,
                    callback=callback
                )
            with self._stream:
                while not self.stop_event.is_set():
                    time.sleep(0.05)
        except sd.CallbackAbort:
            # 預期的中止
            pass
        except Exception as e:
            print(f"[SoundPlayer] Error during playback: {e}")

    def start(self):
        """開始無縫循環播放（直到 stop）"""
        if self.data is None:
            print("[SoundPlayer] No audio loaded; cannot start.")
            return
        if self._thread and self._thread.is_alive():
            return
        # 清旗標
        self.stop_event.clear()
        self._fade_req = False
        self._fade_samples_left = 0

        print("[SoundPlayer] Starting looping playback...")
        self._thread = threading.Thread(target=self._loop_runner, daemon=True)
        self._thread.start()

    def stop(self, *, fade_out_ms: int = 0):
        """停止循環播放；可選擇淡出毫秒數。"""
        if not (self._thread and self._thread.is_alive()):
            return

        if fade_out_ms and self.fs:
            # 觸發淡出，由 callback 逐步把輸出乘權重，然後中止
            self._fade_req = True
            self._fade_samples_left = int(self.fs * (fade_out_ms / 1000.0))
        else:
            self.stop_event.set()

        # 關流與執行緒
        if self._thread:
            self._thread.join(timeout=1.0)

        with self._lock:
            if self._stream:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None

        print("[SoundPlayer] Looping playback stopped.")
