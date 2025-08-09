# utils/sound_player.py

import sounddevice as sd
import soundfile as sf
import threading
import time


class SoundPlayer:
    def __init__(self, sound_file_path):
        self.sound_file_path = sound_file_path
        try:
            # 讀取音訊數據和它的原始取樣率
            self.data, self.fs = sf.read(self.sound_file_path, dtype='float32')
            print(
                f"[SoundPlayer] Loaded '{sound_file_path}' with sample rate {self.fs}Hz.")
        except Exception as e:
            print(f"Error loading sound file {self.sound_file_path}: {e}")
            self.data = None
            return

        self.stop_event = threading.Event()
        self.thread = None
        self.stream = None
        self.lock = threading.Lock()

    def _loop_playback(self):
        """在背景執行緒中循環播放音效，使用明確的 OutputStream。"""
        while not self.stop_event.is_set():
            if self.data is not None:
                try:
                    # [MODIFIED] 使用更可靠的 OutputStream 進行播放
                    current_position = 0
                    playback_finished = threading.Event()

                    def callback(outdata, frames, time, status):
                        nonlocal current_position
                        if status:
                            print(f"[SoundPlayer] Playback status: {status}")

                        chunk_size = len(outdata)
                        remaining = len(self.data) - current_position

                        if remaining >= chunk_size:
                            outdata[:] = self.data[current_position: current_position +
                                                   chunk_size].reshape(-1, 1)
                            current_position += chunk_size
                        elif remaining > 0:
                            outdata[:remaining] = self.data[current_position:].reshape(
                                -1, 1)
                            outdata[remaining:].fill(0)
                            current_position += remaining
                        else:
                            outdata.fill(0)
                            playback_finished.set()

                    with self.lock:
                        # 建立一個新的、參數明確的音訊流
                        self.stream = sd.OutputStream(
                            samplerate=self.fs,  # 強制使用檔案的原始取樣率
                            channels=1,
                            dtype='float32',
                            callback=callback,
                            finished_callback=playback_finished.set
                        )

                    with self.stream:
                        while not playback_finished.is_set() and not self.stop_event.is_set():
                            playback_finished.wait(timeout=0.1)

                except Exception as e:
                    print(f"[SoundPlayer] Error during playback: {e}")
                    # 發生錯誤時等待一秒，避免瘋狂重試
                    time.sleep(1)

    def start(self):
        """開始循環播放"""
        if self.thread and self.thread.is_alive():
            return

        print("[SoundPlayer] Starting looping playback...")
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._loop_playback, daemon=True)
        self.thread.start()

    def stop(self):
        """停止循環播放"""
        if self.thread and self.thread.is_alive():
            print("[SoundPlayer] Stopping looping playback...")
            self.stop_event.set()

            with self.lock:
                if self.stream:
                    self.stream.stop()
                    self.stream.close()

            self.thread.join(timeout=1.0)
            print("[SoundPlayer] Looping playback stopped.")
