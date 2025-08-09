import webrtcvad
import numpy as np
from collections import deque


class WebRTCVAD:
    """
    一個封裝了 Google WebRTC VAD 的類別，用於處理即時音訊流。
    """

    def __init__(self, sample_rate=16000, frame_duration_ms=30, aggressiveness=3):
        """
        初始化 WebRTC VAD。

        Args:
            sample_rate (int): 音訊取樣率 (只支援 8000, 16000, 32000, 48000)。
            frame_duration_ms (int): 每一幀的音訊長度 (只支援 10, 20, 30)。
            aggressiveness (int): VAD 的過濾攻擊性等級 (0-3)，數字越高越嚴格。
        """
        if sample_rate not in [8000, 16000, 32000, 48000]:
            raise ValueError(
                "WebRTC VAD unsupported sample rate: {}".format(sample_rate))
        if frame_duration_ms not in [10, 20, 30]:
            raise ValueError(
                "WebRTC VAD unsupported frame duration: {}".format(frame_duration_ms))

        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_samples = int(sample_rate * frame_duration_ms / 1000)
        self.frame_bytes = self.frame_samples * 2  # 16-bit PCM is 2 bytes per sample

        self.vad = webrtcvad.Vad(aggressiveness)

        # 用於處理不足一幀的音訊數據的內部緩衝區
        self.internal_buffer = bytearray()

    def is_speech(self, audio_bytes: bytes) -> bool:
        """
        判斷一幀音訊是否為人聲。

        Args:
            audio_bytes (bytes): 長度必須等於 frame_bytes 的 16-bit PCM 音訊數據。

        Returns:
            bool: 如果是人聲則為 True，否則為 False。
        """
        if len(audio_bytes) != self.frame_bytes:
            raise ValueError(
                f"Audio chunk must be exactly {self.frame_bytes} bytes for a {self.frame_duration_ms}ms frame.")

        try:
            return self.vad.is_speech(audio_bytes, self.sample_rate)
        except Exception as e:
            print(f"Error during VAD processing: {e}")
            return False

    def process_chunk(self, audio_chunk: np.ndarray) -> list:
        """
        處理一個任意長度的音訊 NumPy 陣列，將其分割成標準幀並進行 VAD 判斷。

        Args:
            audio_chunk (np.ndarray): float32 格式的音訊數據。

        Returns:
            list: 一個包含 (is_speech, frame_bytes) 元組的列表。
                  is_speech 是布林值，frame_bytes 是標準長度的音訊幀。
        """
        # 將 float32 轉為 int16 bytes
        audio_int16 = (audio_chunk * 32767).astype(np.int16)
        self.internal_buffer.extend(audio_int16.tobytes())

        frames = []
        while len(self.internal_buffer) >= self.frame_bytes:
            frame_data = self.internal_buffer[:self.frame_bytes]
            self.internal_buffer = self.internal_buffer[self.frame_bytes:]

            is_speech = self.is_speech(frame_data)
            frames.append((is_speech, frame_data))

        return frames
