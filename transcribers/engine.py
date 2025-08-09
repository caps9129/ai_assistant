import time
import numpy as np
import io
import wave
import math
from abc import ABC, abstractmethod
from faster_whisper import WhisperModel
import speech_recognition as sr
from openai import OpenAI
import logging
from dotenv import load_dotenv
load_dotenv()


class Transcriber(ABC):
    """
    抽象基礎類別，定義了所有語音辨識引擎的通用介面。
    """
    @abstractmethod
    def transcribe(self, audio_bytes: bytes) -> dict:
        """
        將音訊位元組轉換為文字。

        Args:
            audio_bytes: 原始的 PCM 音訊數據。

        Returns:
            一個包含辨識結果的字典，格式為：
            {'text': '辨識的文字', 'error': None 或 '錯誤訊息'}
        """
        pass


class FasterWhisperTranscriber(Transcriber):
    """
    使用 faster-whisper 模型進行離線辨識
    """

    def __init__(self, model_size="medium", device="cpu", compute_type="int8"):

        print(
            f"Loading Whisper model: {model_size} ({device}, {compute_type})...")
        self.model = WhisperModel(
            model_size, device=device, compute_type=compute_type)
        print("✅ Whisper model loaded.")

    def transcribe(self, audio_bytes: bytes) -> dict:
        """
        使用 Whisper 模型進行辨識。
        """
        start_time = time.time()

        # 1. 將 int16 的 bytes 轉換為 float32 的 numpy array
        pcm = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_np = pcm.astype(np.float32) / 32768.0

        try:
            # 2. 進行辨識
            segments, info = self.model.transcribe(audio_np, beam_size=5)
            user_text = "".join(s.text for s in segments).strip()

            # 3. 準備回傳結果
            processing_time = time.time() - start_time
            print(
                f"Whisper transcribed in {processing_time:.2f}s: '{user_text}'")

            return {'text': user_text, 'error': None}

        except Exception as e:
            return {'text': '', 'error': f"Whisper transcription failed: {e}"}

    def detect_language(self, audio_bytes: bytes, duration_s: float = 7.0) -> dict:
        """
        Detects the language from a short, fixed-length chunk of the audio.
        This is much faster than a full transcription or even processing a full sentence.

        Args:
            audio_bytes: Raw PCM audio data for the full utterance.
            duration_s (float): The duration in seconds from the beginning of the
                                audio to use for detection. Defaults to 2.0.

        Returns:
            A dictionary containing the detected language and probability.
        """
        # 1. Convert the full audio to a numpy array
        pcm = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_np = pcm.astype(np.float32) / 32768.0

        # --- NEW: Slice the audio to get a shorter chunk for detection ---
        # We need the sample rate, assuming 16000 as is standard for these models.
        sample_rate = 16000
        num_samples = int(duration_s * sample_rate)
        detection_chunk_np = audio_np[:num_samples]

        print(f"Detecting language from the first {duration_s}s of audio...")

        try:
            # 2. Use the shorter chunk to detect the language
            lang_code, lang_prob, _ = self.model.detect_language(
                audio=detection_chunk_np)

            result = {
                'language': lang_code,
                'probability': lang_prob,
                'error': None
            }
            print(
                f"Detected language: '{result['language']}' with probability {result['probability']:.2f}")
            return result

        except Exception as e:
            return {
                'language': None,
                'probability': 0.0,
                'error': f"Language detection failed: {e}"
            }


class GoogleAPITranscriber(Transcriber):
    """
    使用 speech_recognition 函式庫與 Google Web Speech API 進行線上辨識。
    """

    def __init__(self, language, sample_rate=16000, sample_width=2):
        """
        初始化 Recognizer。
        """
        self.recognizer = sr.Recognizer()
        self.language = language.value
        self.sample_rate = sample_rate
        self.sample_width = sample_width  # 2 bytes for int16
        print(
            f"✅ Google API Transcriber initialized for language: {self.language}.")

    def transcribe(self, audio_bytes: bytes) -> dict:
        """
        使用 Google API 進行辨識。
        """
        if not audio_bytes:
            return {'text': '', 'error': 'Input audio was empty.'}

        # 1. 將原始 bytes 包裝成 AudioData 物件
        audio_data = sr.AudioData(
            audio_bytes, self.sample_rate, self.sample_width)

        try:
            # 2. 呼叫 Google API
            print("... Transcribing with Google Speech Recognition")
            user_text = self.recognizer.recognize_google(
                audio_data,
                language=self.language
            )
            print(f"Google API transcribed: '{user_text}'")
            return {'text': user_text, 'error': None}

        except sr.UnknownValueError:
            return {'text': '', 'error': 'Google could not understand the audio.'}
        except sr.RequestError as e:
            return {'text': '', 'error': f'Google API request failed: {e}'}


class OpenAITranscriber(Transcriber):
    """
    使用 OpenAI 的多语种转写模型，并在关键步骤输出调试信息。
    """

    def __init__(self, model: str = "gpt-4o-mini-transcribe", noise_threshold: float = 0.7):
        self.client = OpenAI()
        self.model_name = model
        self.noise_threshold = noise_threshold
        logging.getLogger("openai").setLevel(logging.INFO)
        logging.info(
            f"✅ Initialized OpenAITranscriber with model={self.model_name}, threshold={self.noise_threshold}")

    def pcm_bytes_to_wav_bytes(self, pcm_bytes: bytes, sample_rate=16000, channels=1, sampwidth=2) -> bytes:
        """
        将 PCM bytes 封装为 WAV bytes，失败时抛出异常。
        """

        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)
        wav_data = buf.getvalue()
        logging.debug(
            f"PCM→WAV 成功: {len(pcm_bytes)}B PCM → {len(wav_data)}B WAV")
        return wav_data

    def transcribe(self, audio_bytes: bytes) -> dict:
        """
        核心转写方法，返回 {'text', 'confidence'} 或 {'text':'', 'reason'} / {'error'}。
        """
        if not audio_bytes:
            logging.warning("输入音频为空")
            return {"text": "", "reason": "empty input"}

        # 1) PCM→WAV
        wav_bytes = self.pcm_bytes_to_wav_bytes(audio_bytes)

        # 2) 准备 BytesIO
        audio_buffer = io.BytesIO(wav_bytes)
        audio_buffer.name = "audio.wav"

        # 3) 调用 API
        try:
            stream = self.client.audio.transcriptions.create(
                model=self.model_name,
                file=audio_buffer,
                response_format="json",
                include=["logprobs"],
                stream=True
            )
        except Exception as e:
            return {"text": "", "error": f"API error: {e}"}

        # 4) 读取直到 transcript.text.done
        done_event = None
        for event in stream:
            if getattr(event, "type", "") == "transcript.text.done":
                done_event = event
                break

        # 5) 检查 DoneEvent 和 logprobs
        if not done_event:
            return {"text": "", "reason": "no done event"}
        if not getattr(done_event, "logprobs", None):
            return {"text": "", "reason": "no logprobs"}

        # 6) 计算平均置信度

        avg_lp = sum(lp.logprob for lp in done_event.logprobs) / \
            len(done_event.logprobs)
        avg_conf = math.exp(avg_lp)

        # 7) 噪音阈值过滤
        if avg_conf < self.noise_threshold:
            logging.info(
                f"confidence score {avg_conf:.2%}")
            return {"text": "", "reason": f"low confidence ({avg_conf:.2%})"}

        # 8) 返回最终结果
        logging.info(f"text: \"{done_event.text}\" (conf {avg_conf:.2%})")
        return {"text": done_event.text, "confidence": avg_conf}


# --- 主程式如何使用 ---
if __name__ == '__main__':
    # 假設這是從你的 AudioSegmenter 獲得的音訊數據
    # 這裡我們建立一個 2 秒的靜音作為範例
    SAMPLE_RATE = 16000
    SILENT_BYTES = b'\x00\x00' * SAMPLE_RATE * 2  # 2 seconds of silence

    print("="*30)
    print("DEMO 1: Using FasterWhisperTranscriber")
    print("="*30)
    # 在你的主程式中，你只需要初始化一次
    whisper_transcriber = FasterWhisperTranscriber("tiny")
    # 傳入靜音，Whisper 應該會回傳空字串
    result_whisper = whisper_transcriber.transcribe(SILENT_BYTES)
    print(f"--> Final Result: {result_whisper}\n")

    print("="*30)
    print("DEMO 2: Using GoogleAPITranscriber")
    print("="*30)
    # 同樣地，只需要初始化一次language="zh-TW"
    from config import Language
    google_transcriber = GoogleAPITranscriber(language=Language.ENGLISH)
    # 傳入靜音，Google API 應該會回傳錯誤

    audio, file_samplerate = sf.read("/home/aiden/Desktop/aiden/test.wav")

    # 3. 確保音訊是單聲道
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # 4. 確保取樣率是 16000 Hz (如果不是，需要重サンプリング)
    # 為了簡化，我們先假設取樣率正確，但這是一個專業應用中需要處理的步驟
    if file_samplerate != 16000:
        print(f"⚠️ 警告：音訊檔案的取樣率是 {file_samplerate} Hz，但模型需要 16000 Hz。結果可能不準確。")
        # 在真實應用中，您會在這裡加入重サンプリング的程式碼
        # import librosa
        # audio = librosa.resample(y=audio, orig_sr=file_samplerate, target_sr=16000)

    # 5. 確保數據類型是 int16
    # 將浮點數音訊轉換為 16-bit 整數
    if audio.dtype == np.float32 or audio.dtype == np.float64:
        audio = (audio * 32767).astype(np.int16)

    # 6. 現在可以安全地轉換為位元組
    audio_bytes = audio.tobytes()

    # 7. 進行轉錄
    result_google = google_transcriber.transcribe(audio_bytes)
    print(f"--> Final Result: {result_google}\n")
