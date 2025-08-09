# audio/pipeline.py

import threading
import queue
import numpy as np
import json
import time
from pathlib import Path

from vad.webrtc import WebRTCVAD
from transcribers.router import AsrRouter


class PerceptionPipeline:
    def __init__(self, asr_router: AsrRouter, on_result_callback):
        # [FIX] 修正 config 檔案的路徑，讓它更準確
        config_path = Path(__file__).parent.parent / "config.json"
        cfg = json.load(config_path.open("r"))
        self.sample_rate = cfg["sample_rate"]
        min_silence_duration_ms = cfg["min_silence_duration_ms"]

        self.asr_router = asr_router
        self.on_result_callback = on_result_callback

        self.vad_engine = WebRTCVAD(
            sample_rate=self.sample_rate, frame_duration_ms=30, aggressiveness=3)
        self.silence_frames_threshold = int(
            min_silence_duration_ms / self.vad_engine.frame_duration_ms)

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

    def _asr_loop(self):
        while self.is_running:
            # [MODIFIED] 增加時間戳以計算延遲
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

            if result and result.get("text", "").strip():
                print(f"[Perception] ASR result is valid: '{result['text']}'")
                self.on_result_callback(result)
            else:
                print("[Perception] ASR result is empty, ignored as noise.")

            # 只有在 ASR 完成後，才真正結束一次處理流程
            self._set_processing_status(False)

    def _process_loop(self):
        """
        [MODIFIED] 這是本次的核心修正：重寫了 VAD 的狀態判斷邏輯，使其更穩健。
        """
        in_speech = False
        audio_buffer = bytearray()
        silent_frames_count = 0

        while self.is_running:

            try:
                with self.status_lock:
                    if self._is_paused:
                        time.sleep(0.1)
                        in_speech = False
                        audio_buffer.clear()
                        continue

                audio_chunk = self.audio_queue.get(timeout=0.2)
                frames = self.vad_engine.process_chunk(audio_chunk)

                for is_speech_frame, frame_bytes in frames:
                    if is_speech_frame:
                        # --- 偵測到語音 ---
                        if not in_speech:
                            print("[Perception] VAD Detected speech start.")
                            in_speech = True
                            self._set_processing_status(True)

                        audio_buffer.extend(frame_bytes)
                        silent_frames_count = 0  # 只要有語音，就重置靜音計數

                    elif in_speech:
                        # --- 偵測到語音後的靜音 ---
                        silent_frames_count += 1
                        if silent_frames_count > self.silence_frames_threshold:
                            vad_end_time = time.time()
                            print(
                                f"[Perception] VAD finalized a segment due to silence at {vad_end_time:.2f}, sending to ASR worker.")
                            self.asr_queue.put(
                                (audio_buffer.copy(), vad_end_time))

                            # 重置狀態，準備下一句話
                            in_speech = False
                            audio_buffer.clear()
                            silent_frames_count = 0
                            # 注意：is_processing 狀態由 ASR worker 結束

            except queue.Empty:
                if in_speech:
                    vad_end_time = time.time()
                    print(
                        f"[Perception] Finalizing segment due to inactivity at {vad_end_time:.2f}, sending to ASR worker.")
                    self.asr_queue.put((audio_buffer.copy(), vad_end_time))
                    in_speech = False
                    audio_buffer.clear()
                    silent_frames_count = 0
                continue

    def process_audio(self, audio_chunk: np.ndarray):
        if self.is_running:
            self.audio_queue.put(audio_chunk)

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.processing_thread.start()
            self.asr_worker_thread.start()
            print("✅ Perception Pipeline started.")

    def stop(self):
        if self.is_running:
            self.is_running = False
            self.asr_queue.put((None, None))
            self.processing_thread.join(timeout=1.0)
            self.asr_worker_thread.join(timeout=1.0)
            print("🛑 Perception Pipeline stopped.")
