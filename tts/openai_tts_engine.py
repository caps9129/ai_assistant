# tts/openai_tts_engine.py

import time
from .base_engine import TTSEngineBase
from queue import Queue, Empty
import threading
import sounddevice as sd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


class OpenAITTSEngine(TTSEngineBase):
    def __init__(self, voice: str, stop_event: threading.Event, model: str = "tts-1", **kwargs):
        super().__init__(voice=voice, stop_event=stop_event)
        self.model = model
        self.client = OpenAI()
        self.sample_rate = 24000  # OpenAI TTS is 24kHz

        self.pcm_data_queue = Queue(maxsize=1)
        self.playback_thread = None
        self.stream_running = threading.Event()

    def _playback_loop(self):
        """在背景持續運行的迴圈，是音訊播放的核心。"""
        current_pcm_data = None
        current_position = 0

        def callback(outdata, frames, time, status):
            nonlocal current_pcm_data, current_position
            if status:
                print(f"[{self.__class__.__name__}] Playback status: {status}")

            if current_pcm_data is None:
                try:
                    item = self.pcm_data_queue.get_nowait()
                    if isinstance(item, np.ndarray):
                        current_pcm_data = item
                        current_position = 0
                    elif item is None:
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
                current_pcm_data = None

        print(f"[{self.__class__.__name__}] Starting persistent audio stream...")
        try:
            with sd.OutputStream(
                samplerate=self.sample_rate, channels=1, dtype='int16', callback=callback
            ) as stream:
                self.stream_running.set()
                self.stop_event.wait()
            print(f"[{self.__class__.__name__}] Persistent audio stream stopped.")
        except Exception as e:
            print(f"[{self.__class__.__name__}] Audio stream error: {e}")
        finally:
            self.stream_running.clear()

    def start_stream(self):
        if self.playback_thread and self.playback_thread.is_alive():
            return
        print(f"[{self.__class__.__name__}] Start stream command received.")
        self.stop_event.clear()
        self.playback_thread = threading.Thread(
            target=self._playback_loop, daemon=True)
        self.playback_thread.start()
        self.stream_running.wait(timeout=5.0)

    def stop_stream(self):
        if not self.playback_thread or not self.playback_thread.is_alive():
            return
        print(f"[{self.__class__.__name__}] Stop stream command received.")
        self.stop_event.set()
        self.pcm_data_queue.put(None)
        self.playback_thread.join(timeout=2.0)
        self.playback_thread = None

    def _fetch_and_decode_audio(self, text: str):
        """獲取完整的 PCM 音訊。"""
        print(f"[{self.__class__.__name__}] Fetching PCM for text: '{text[:20]}...'")
        response = self.client.audio.speech.create(
            model=self.model, voice=self.voice, response_format="pcm", input=text
        )

        # 預先拼接靜音
        # silence = np.zeros(int(self.sample_rate * 0.1), dtype=np.int16)
        pcm_data = np.frombuffer(response.content, dtype=np.int16)

        # return np.concatenate([silence, pcm_data])
        return pcm_data

    def say(self, text: str):
        if not self.stream_running.is_set():
            print(f"[{self.__class__.__name__}] ERROR: Stream is not running.")
            return

        print(
            f"[{self.__class__.__name__}] Servicing TTS request for: '{text[:20]}...'")

        try:
            # --- DEBUG 點 1: 標記異步獲取開始 ---
            print("[DEBUG] say: Starting _fetch_and_decode_audio...")
            pcm_data = self._fetch_and_decode_audio(text)

            if pcm_data is not None:
                print(
                    f"[DEBUG] say: Audio data received. Shape: {pcm_data.shape}, a non-blocking put to the queue will be performed")

                self.pcm_data_queue.put(pcm_data)

                expected_duration = len(pcm_data) / self.sample_rate
                wait_time = expected_duration + 0.5
                print(
                    f"[{self.__class__.__name__}] Audio queued. Waiting for playback to finish (approx. {wait_time:.2f}s)...")

                # --- DEBUG 點 3: 標記等待迴圈開始 ---
                print("[DEBUG] say: Entering wait loop...")
                start_time = time.time()
                while time.time() - start_time < wait_time:
                    if self.stop_event.is_set():
                        print(
                            f"[{self.__class__.__name__}] Playback interrupted by stop event.")
                        break
                    time.sleep(0.1)

                # --- DEBUG 點 4: 標記等待迴圈正常結束 ---
                print("[DEBUG] say: Wait loop finished.")

            else:
                # --- DEBUG 點 5: 標記音訊獲取失敗 ---
                print(
                    "[DEBUG] say: _fetch_and_decode_audio returned None. No audio to play.")

        except Exception as e:
            print(f"[{self.__class__.__name__}] Error in say method: {e}")

        print(f"[{self.__class__.__name__}] SAY method finished for this text.")

    def stop(self):
        print(f"[{self.__class__.__name__}] STOP command received, clearing queue.")
        while not self.pcm_data_queue.empty():
            self.pcm_data_queue.get_nowait()
        self.stop_event.set()
