# tts/edge_tts_engine.py

import asyncio
from io import BytesIO
import edge_tts
import numpy as np
import sounddevice as sd
from pydub import AudioSegment
import threading
from queue import Queue, Empty
from .base_engine import TTSEngineBase
import time


class EdgeTTSEngine(TTSEngineBase):
    def __init__(self, voice: str, stop_event: threading.Event, sample_rate: int = 24000, **kwargs):
        super().__init__(voice=voice, stop_event=stop_event)
        self.target_sample_rate = sample_rate

        self.pcm_data_queue = Queue(maxsize=1)
        self.playback_thread = None
        self.stream_running = threading.Event()

    def _playback_loop(self):
        """在背景持續運行的迴圈，是音訊播放的核心。"""
        current_pcm_data = None
        current_position = 0
        playback_finished_for_current_audio = threading.Event()

        def callback(outdata, frames, time, status):
            nonlocal current_pcm_data, current_position
            if status:
                print(f"[{self.__class__.__name__}] Playback status: {status}")

            if current_pcm_data is None:
                try:
                    # 嘗試從佇列中獲取新的、完整的音訊
                    item = self.pcm_data_queue.get_nowait()
                    if isinstance(item, np.ndarray):
                        current_pcm_data = item
                        current_position = 0
                        playback_finished_for_current_audio.clear()
                    elif item is None:  # 收到結束信號
                        raise sd.CallbackStop
                except Empty:
                    outdata.fill(0)
                    return

            chunk_size = len(outdata)
            remaining_data = len(current_pcm_data) - current_position

            if remaining_data >= chunk_size:
                outdata[:] = current_pcm_data[current_position: current_position +
                                              chunk_size].reshape(-1, 1)
                current_position += chunk_size
            else:
                if remaining_data > 0:
                    outdata[:remaining_data] = current_pcm_data[current_position:].reshape(
                        -1, 1)
                    outdata[remaining_data:] = 0
                else:
                    outdata.fill(0)

                # 當前音訊已播放完畢
                current_pcm_data = None
                playback_finished_for_current_audio.set()

        print(f"[{self.__class__.__name__}] Starting persistent audio stream...")
        try:
            with sd.OutputStream(
                samplerate=self.target_sample_rate, channels=1, dtype='int16', callback=callback
            ) as stream:
                self.stream_running.set()
                self.stop_event.wait()
            print(f"[{self.__class__.__name__}] Persistent audio stream stopped.")
        except Exception as e:
            print(f"[{self.__class__.__name__}] Audio stream error: {e}")
        finally:
            self.stream_running.clear()

    def start_stream(self):
        # ... (此方法保持不變) ...
        if self.playback_thread and self.playback_thread.is_alive():
            return
        print(f"[{self.__class__.__name__}] Start stream command received.")
        self.stop_event.clear()
        self.playback_thread = threading.Thread(
            target=self._playback_loop, daemon=True)
        self.playback_thread.start()
        self.stream_running.wait(timeout=5.0)

    def stop_stream(self):
        # ... (此方法保持不變) ...
        if not self.playback_thread or not self.playback_thread.is_alive():
            return
        print(f"[{self.__class__.__name__}] Stop stream command received.")
        self.stop_event.set()
        self.pcm_data_queue.put(None)  # 放入結束信號以喚醒 get()
        self.playback_thread.join(timeout=2.0)
        self.playback_thread = None

    async def _fetch_and_decode_audio(self, text: str):
        # ... (此方法保持不變) ...
        print(f"[{self.__class__.__name__}] Fetching MP3 for text: '{text[:20]}...'")
        communicator = edge_tts.Communicate(text, self.voice)
        mp3_buffer = bytearray()
        async for chunk in communicator.stream():
            if chunk["type"] == "audio":
                mp3_buffer.extend(chunk["data"])
        audio = AudioSegment.from_file(BytesIO(mp3_buffer), format="mp3")
        audio = audio.set_frame_rate(
            self.target_sample_rate).set_channels(1).set_sample_width(2)
        silence = np.zeros(int(self.target_sample_rate * 0.1), dtype=np.int16)
        pcm_data = np.frombuffer(audio.raw_data, dtype=np.int16)
        return np.concatenate([silence, pcm_data])

    def say(self, text: str):
        if not self.stream_running.is_set():
            print(f"[{self.__class__.__name__}] ERROR: Stream is not running.")
            return

        print(
            f"[{self.__class__.__name__}] Servicing TTS request for: '{text[:20]}...'")

        # [MODIFIED] say 現在是阻塞的，它會等待音訊被獲取並放入佇列
        try:
            pcm_data = asyncio.run(self._fetch_and_decode_audio(text))
            if pcm_data is not None:
                self.pcm_data_queue.put(pcm_data)

                # 等待，直到 playback_loop 確認播放完畢
                # 這裡需要一個方法來監控，我們簡化一下，
                # 透過計算音訊長度來進行等待
                expected_duration = len(pcm_data) / self.target_sample_rate

                # 給予一個緩衝時間
                wait_time = expected_duration + 0.5
                print(
                    f"[EdgeTTSEngine] Audio queued. Waiting for playback to finish (approx. {wait_time:.2f}s)...")

                # 簡易的等待機制，更複雜的需要雙向事件通信
                start_time = time.time()
                while time.time() - start_time < wait_time:
                    if self.stop_event.is_set():
                        print("[EdgeTTSEngine] Playback interrupted by stop event.")
                        break
                    time.sleep(0.1)

        except Exception as e:
            print(f"[{self.__class__.__name__}] Error in say method: {e}")

        print(f"[{self.__class__.__name__}] SAY method finished for this text.")

    def stop(self):
        print(f"[{self.__class__.__name__}] STOP command received, clearing queue.")
        while not self.pcm_data_queue.empty():
            self.pcm_data_queue.get_nowait()
        self.stop_event.set()  # 呼叫 stop_event 以停止持久流
