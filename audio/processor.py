# audio/processor.py

import torch
import sounddevice as sd
import numpy as np
import threading
from openwakeword.model import Model as WakeWordModel
from pathlib import Path
import json


class VoiceProcessor:
    # [MODIFIED] 移除了 on_sentence 參數，因為這個職責已轉交給 PerceptionPipeline
    def __init__(self, on_wakeword, perception_pipeline=None):
        # 回调
        self.on_wakeword = on_wakeword
        self.perception_pipeline = perception_pipeline

        # 线程安全锁
        self.lock = threading.Lock()

        # 状态：IDLE / DIALOG
        self.state = "IDLE"
        self.running = False
        self.stream = None

        # [REMOVED] 不再需要 is_interruption_enabled，因為 processor 不再關心打斷邏輯
        # self.is_interruption_enabled = False

        # 加载音频配置
        cfg_path = Path(__file__).parent / "config.json"
        cfg = json.load(open(cfg_path, "r"))
        self.sample_rate = cfg["sample_rate"]
        self.frame_duration_ms = cfg["frame_duration_ms"]

        # WakeWord 模型
        project_root = Path(__file__).parent.parent
        ww_cfg = json.load(
            open(project_root / "wakeword" / "config.json", "r"))
        self.wakeword_threshold = ww_cfg["threshold"]
        self.wakeword_model = WakeWordModel(
            wakeword_models=[str(project_root / ww_cfg["model"])],
            melspec_model_path=str(project_root / ww_cfg["melspec_model"]),
            embedding_model_path=str(project_root / ww_cfg["embedding_model"]),
            inference_framework="onnx",
        )

    def _audio_callback(self, indata, frames, time, status):
        if not self.running:
            return
        if status:
            print(f"[Callback] Stream status: {status}")

        audio_int16 = indata[:, 0]

        with self.lock:
            state = self.state

        if state == "IDLE":
            self.wakeword_model.predict(audio_int16)
            for buf in self.wakeword_model.prediction_buffer.values():
                if buf[-1] > self.wakeword_threshold:
                    self.on_wakeword()
                    break
        elif state == "DIALOG":
            # 在 DIALOG 狀態，所有音訊都交給 perception_pipeline
            if self.perception_pipeline:
                audio_float32 = audio_int16.astype(np.float32) / 32768.0
                self.perception_pipeline.process_audio(audio_float32)

    def set_state(self, new_state):
        with self.lock:
            if self.state == new_state:
                return
            print(f"[Processor] State {self.state} → {new_state}")
            self.state = new_state
        self.wakeword_model.reset()

    def start(self):
        self.running = True
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=int(self.sample_rate * self.frame_duration_ms / 1000),
            callback=self._audio_callback,
        )
        self.stream.start()
        print("✅ VoiceProcessor started.")

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.running = False
        print("🛑 VoiceProcessor stopped.")
