import torch
import numpy as np


class SileroVAD:
    """
    一個封裝了 Silero VAD 的類別，現在支援 24000Hz。
    """

    def __init__(self, sample_rate=24000, threshold=0.5):
        if sample_rate not in [8000, 16000]:
            # Silero VAD 官方模型只支援 8k 和 16k，但它在實踐中對更高的取樣率有不錯的兼容性
            # 我們將在 process_chunk 中處理這個問題
            print(
                f"Warning: Silero VAD officially supports 8k/16k. You are using {sample_rate}Hz.")

        self.target_sample_rate = sample_rate
        self.model, self.utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True
        )
        self.vad_iterator = self.utils[3](self.model, threshold=threshold)

        # Silero VAD 期望的內部處理取樣率是 16000Hz
        self.internal_sample_rate = 16000

    def process_chunk(self, audio_chunk: np.ndarray) -> bool:
        """
        處理一個音訊塊，判斷其中是否包含語音。
        如果輸入取樣率不是 16k，會進行重採樣。

        Args:
            audio_chunk (np.ndarray): float32 格式的音訊數據。

        Returns:
            bool: 如果偵測到語音則為 True。
        """
        if self.target_sample_rate != self.internal_sample_rate:

            import resampy
            audio_chunk = resampy.resample(
                audio_chunk,
                self.target_sample_rate,
                self.internal_sample_rate
            )

        speech_dict = self.vad_iterator(torch.from_numpy(audio_chunk))
        if speech_dict and ("start" in speech_dict or "end" in speech_dict):
            return True
        return False

    def reset_states(self):
        self.vad_iterator.reset_states()
