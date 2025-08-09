# tts/router.py

import fasttext
from pathlib import Path
from tts.engine import TTSEngine
import threading
import nltk
nltk.data.find('tokenizers/punkt')


class FastTextLangDetector:
    def __init__(self):
        try:
            script_dir = Path(__file__).parent
            model_path = script_dir / 'models' / 'lid.176.bin'
            self.model = fasttext.load_model(str(model_path))
            print("✅ fastText model loaded successfully.")
        except ValueError as e:
            print(f"❌ Error loading fastText model: {e}")
            raise

    def detect(self, text: str) -> tuple[str, float]:
        text = text.replace("\n", " ")
        predictions = self.model.predict(text, k=1)
        lang_code = predictions[0][0].replace('__label__', '')
        confidence = predictions[1][0]
        return lang_code, confidence


class LanguageRouterTTS:
    def __init__(self, stop_event: threading.Event):
        print("Initializing LanguageRouterTTS...")
        self.detector = FastTextLangDetector()
        self.stop_event = stop_event  # [NEW] 保存 stop_event
        self._initialize_engines()

    def _initialize_engines(self):
        print("... Initializing TTS engines...")
        TTS_SAMPLE_RATE = 24000
        # [MODIFIED] 將 stop_event 傳遞給 TTSEngine
        self.tts_en = TTSEngine(
            voice="en-US-AriaNeural", sample_rate=TTS_SAMPLE_RATE, stop_event=self.stop_event)
        self.tts_zh = TTSEngine(voice="zh-TW-HsiaoChenNeural",
                                sample_rate=TTS_SAMPLE_RATE, stop_event=self.stop_event)

    def say(self, text: str):
        lang_code, confidence = self.detector.detect(text)
        print(
            f"Detected language for TTS: {lang_code} with confidence {confidence:.2f}")
        if lang_code == 'zh':
            self.tts_zh.say(text)
        else:
            self.tts_en.say(text)

    def say_stream(self, text_generator):
        """
        [MODIFIED]
        使用 NLTK 專業斷句函式庫來處理文字流。
        """
        print("[TTS Router] Starting to process text stream with NLTK...")
        text_buffer = ""

        for text_chunk in text_generator:
            if self.stop_event.is_set():
                print("[TTS Router] Stop event detected, stopping stream playback.")
                break

            text_buffer += text_chunk

            # 使用 NLTK 進行斷句
            sentences = nltk.sent_tokenize(text_buffer)

            # NLTK 會將所有句子都分割出來。
            # 如果 text_buffer 的結尾不是一個完整的句子，
            # 最後一個元素就會是那段不完整的句子。
            if len(sentences) > 1:
                # 播放除了最後一個（可能不完整）之外的所有句子
                for sentence in sentences[:-1]:
                    if sentence.strip():
                        print(
                            f"[TTS Router] Speaking sentence: '{sentence.strip()}'")
                        self.say(sentence.strip())
                        if self.stop_event.is_set():
                            break  # 檢查是否在播放期間被打斷

                if self.stop_event.is_set():
                    break

                # 將最後一個不完整的句子放回緩衝區，等待下一個 chunk
                text_buffer = sentences[-1]

        # 處理最後剩餘的文字
        if text_buffer.strip() and not self.stop_event.is_set():
            print(
                f"[TTS Router] Speaking final sentence: '{text_buffer.strip()}'")
            self.say(text_buffer.strip())

        print("[TTS Router] Text stream finished.")

    def stop(self):
        # [MODIFIED] stop 現在只設定事件
        self.stop_event.set()

    def reset(self):
        print("[TTS Router] Resetting engines...")
        self._initialize_engines()
