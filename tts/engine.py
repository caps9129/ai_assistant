import asyncio
from io import BytesIO
import edge_tts
import numpy as np
import sounddevice as sd
from pydub import AudioSegment
import threading


class TTSEngine:
    def __init__(self, voice: str, sample_rate: int, stop_event: threading.Event):
        self.voice = voice
        self.target_sample_rate = sample_rate
        self.stop_event = stop_event  # [NEW] 接收外部的停止信號

        self.stream = None
        self.playback_finished = threading.Event()
        self.pcm_data = None
        self.current_position = 0
        self.lock = threading.Lock()
        print(
            f"[TTS Engine] Initialized for voice {voice} with target sample rate {self.target_sample_rate}Hz")

    async def _fetch_full_mp3(self, text: str) -> bytes:

        print(f"[TTS Engine] Fetching MP3 for text: '{text[:20]}...'")
        communicator = edge_tts.Communicate(text, self.voice)
        buf = bytearray()
        async for chunk in communicator.stream():
            if chunk.get("type") == "audio":
                buf.extend(chunk.get("data", b""))
        print("[TTS Engine] MP3 fetch complete.")
        return bytes(buf)

    def _mp3_to_pcm(self, mp3_bytes: bytes) -> np.ndarray:

        audio = AudioSegment.from_file(BytesIO(mp3_bytes), format="mp3")
        audio = audio.set_frame_rate(
            self.target_sample_rate).set_channels(1).set_sample_width(2)
        return np.frombuffer(audio.raw_data, dtype='<i2')

    def _playback_callback(self, outdata, frames, time, status):

        if status:
            print(f"[TTS Engine] Playback Status Alert: {status}")
        with self.lock:
            chunk_size = len(outdata)
            remaining_data = len(self.pcm_data) - self.current_position
            if remaining_data >= chunk_size:
                outdata[:] = self.pcm_data[self.current_position:
                                           self.current_position + chunk_size].reshape(-1, 1)
                self.current_position += chunk_size
            elif remaining_data > 0:
                outdata[:remaining_data] = self.pcm_data[self.current_position:].reshape(
                    -1, 1)
                outdata[remaining_data:] = 0
                self.current_position += remaining_data
            else:
                outdata.fill(0)
                if not self.playback_finished.is_set():
                    self.playback_finished.set()

    def say(self, text: str):
        try:
            mp3_data = asyncio.run(self._fetch_full_mp3(text))
            if not mp3_data:
                return

            self.pcm_data = self._mp3_to_pcm(mp3_data)

            with self.lock:
                self.current_position = 0
                self.playback_finished.clear()

            self.stream = sd.OutputStream(
                samplerate=self.target_sample_rate, channels=1, dtype='int16',
                callback=self._playback_callback, finished_callback=self.playback_finished.set
            )
            with self.stream:
                # [MODIFIED] 不再是簡單的 wait()，而是一個可以監聽 stop_event 的迴圈
                while not self.playback_finished.is_set():
                    if self.stop_event.is_set():
                        print(
                            "[TTS Engine] Stop event detected inside say(), stopping stream.")
                        self.stream.stop()
                        self.stream.close()
                        break
                    self.playback_finished.wait(timeout=0.1)  # 短暫等待

        except Exception as e:
            print(f"[TTS Engine] ERROR in say(): {e}")
        finally:
            if not self.playback_finished.is_set():
                self.playback_finished.set()
            print("[TTS Engine] SAY method finished.")

    def stop(self):
        # [MODIFIED] stop 的職責現在極其簡單：只設定信號
        print("[TTS Engine] STOP command received, setting stop event.")
        self.stop_event.set()

    def start_stream(self):
        """啟動所有引擎的持久音訊流"""
        print("[TTS Router] Starting persistent streams for all engines...")
        for engine in self.tts_engines.values():
            if hasattr(engine, 'start_stream'):
                engine.start_stream()

    def stop_stream(self):
        """停止所有引擎的持久音訊流"""
        print("[TTS Router] Stopping persistent streams for all engines...")
        for engine in self.tts_engines.values():
            if hasattr(engine, 'stop_stream'):
                engine.stop_stream()
