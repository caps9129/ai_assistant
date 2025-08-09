# tts/router.py

import threading
import time
from pathlib import Path
import fasttext
import nltk

# 導入我們設計的設定檔和 TTS 引擎
# 假設這些檔案都已存在於 tts/ 資料夾中
from .config import TTS_CONFIG
from .base_engine import TTSEngineBase
from .edge_tts_engine import EdgeTTSEngine
from .openai_tts_engine import OpenAITTSEngine

# 確保 NLTK 的斷句模型存在，如果不存在則自動下載
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("NLTK 'punkt' model not found. Downloading...")
    nltk.download('punkt', quiet=True)


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
        text = text.replace("\n", " ").strip()
        if not text:
            return "en", 0.9  # 對於空字串，預設為英文
        predictions = self.model.predict(text, k=1)
        lang_code = predictions[0][0].replace('__label__', '')
        confidence = predictions[1][0]
        return lang_code, confidence


class LanguageRouterTTS:
    def __init__(self, stop_event: threading.Event = None, tts_provider: str = None):
        print("Initializing LanguageRouterTTS...")

        self.stop_event = stop_event if stop_event else threading.Event()
        self.detector = FastTextLangDetector()

        self.config = TTS_CONFIG
        self.active_provider = tts_provider or self.config.get(
            "active_provider", "edge")
        print(f"... Active TTS provider: {self.active_provider}")

        self.tts_engines: dict[str, TTSEngineBase] = {}
        self._initialize_engines()

    def _initialize_engines(self):
        """根據 config 初始化對應的 TTS 引擎"""
        provider_config = self.config["providers"][self.active_provider]
        EngineClass = EdgeTTSEngine if self.active_provider == "edge" else OpenAITTSEngine

        for lang, config in provider_config.items():
            print(
                f"... Initializing '{lang}' voice using {self.active_provider}...")
            # 使用 kwargs 以獲得更好的彈性
            self.tts_engines[lang] = EngineClass(
                voice=config.voice,
                stop_event=self.stop_event,
                model=config.model,
                sample_rate=config.sample_rate
            )

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

    def say(self, text: str):
        """播放單一、完整的文字字串。"""
        if not text or not text.strip():
            return

        engine = None
        if self.active_provider == "openai":
            # OpenAI 不分語言，隨便選一個引擎即可
            engine = self.tts_engines.get("en")
        else:
            lang_code, _ = self.detector.detect(text)
            # 選擇對應語言的引擎，如果不存在則 fallback 到英文
            engine = self.tts_engines.get(
                lang_code, self.tts_engines.get("en"))

        if engine:
            engine.say(text)
        else:
            print(
                f"ERROR: Could not find a suitable TTS engine for provider '{self.active_provider}'")

    def say_stream(self, text_generator):
        """使用 NLTK 專業斷句函式庫來處理文字流並逐句播放。"""
        print("\n[TTS Router] Starting to process text stream with NLTK...")
        text_buffer = ""

        for text_chunk in text_generator:
            if self.stop_event.is_set():
                print("[TTS Router] Stop event detected, stopping stream playback.")
                break

            text_buffer += text_chunk
            sentences = nltk.sent_tokenize(text_buffer)

            if len(sentences) > 1:
                for sentence in sentences[:-1]:
                    sentence = sentence.strip()
                    if sentence:
                        print(f"[TTS Router] Speaking sentence: '{sentence}'")
                        self.say(sentence)
                        if self.stop_event.is_set():
                            break

                if self.stop_event.is_set():
                    break
                text_buffer = sentences[-1]

        if text_buffer.strip() and not self.stop_event.is_set():
            print(
                f"[TTS Router] Speaking final sentence: '{text_buffer.strip()}'")
            self.say(text_buffer.strip())

        print("[TTS Router] Text stream finished.")

    def stop(self):
        self.stop_event.set()

    def reset(self):
        print("[TTS Router] Resetting engines...")
        self.stop_stream()
        self._initialize_engines()
        self.start_stream()


# --- 測試方法 ---

def simulate_llm_stream(text, chunk_size=10, delay=0.05):
    """一個產生器函式，用於模擬 LLM 的串流輸出效果。"""
    print(
        f"\n--- Simulating LLM Stream (chunk_size={chunk_size}, delay={delay}s) ---")
    for i in range(0, len(text), chunk_size):
        yield text[i:i+chunk_size]
        time.sleep(delay)
    print("--- LLM Stream Simulation Finished ---")


if __name__ == '__main__':
    # 這個區塊讓您可以直接運行 `python -m tts.router` 來進行獨立測試
    # openai
    PROVIDER_TO_TEST = "openai"
    tts_router = LanguageRouterTTS(tts_provider=PROVIDER_TO_TEST)

    # 2. 準備測試文字
    full_text = "這是一個完整的句子，用於測試非串流播放。"
    stream_text = "這是一個串流測試。第一句話會被先播放。這是第二句，它會稍微晚一點出現。最後，才是結尾的部分。"

    # 3. 測試持久流的生命週期 (Start/Stop Stream)
    print("\n" + "="*40)
    print(">>> 測試 1: 持久流生命週期 (start_stream / stop_stream)")
    print("="*40)
    tts_router.start_stream()
    print("... 持久流已啟動，等待 2 秒...")
    time.sleep(2)
    tts_router.stop_stream()
    print("... 持久流已停止。")

    # 4. 測試非串流 (Non-Stream) 播放
    print("\n" + "="*40)
    print(">>> 測試 2: 非串流 say() 方法")
    print("="*40)
    tts_router.start_stream()  # 播放前需要啟動流
    tts_router.say(full_text)
    tts_router.stop_stream()  # 播放後關閉流

    # 5. 測試串流 (Stream) 播放
    print("\n" + "="*40)
    print(">>> 測試 3: 串流 say_stream() 方法")
    print("="*40)
    tts_router.start_stream()
    llm_generator = simulate_llm_stream(stream_text)
    tts_router.say_stream(llm_generator)
    tts_router.stop_stream()

    # 6. 測試打斷功能
    print("\n" + "="*40)
    print(">>> 測試 4: 串流播放時的打斷功能")
    print("="*40)

    def interrupt_after_delay(delay, router):
        time.sleep(delay)
        print("\n*** [TEST] 發送停止信號! ***\n")
        router.stop()

    tts_router.start_stream()
    interrupt_thread = threading.Thread(
        target=interrupt_after_delay, args=(5, tts_router))
    llm_generator_interrupt = simulate_llm_stream(stream_text)

    interrupt_thread.start()
    tts_router.say_stream(llm_generator_interrupt)
    # 打斷後，流可能還在運行，我們需要手動停止它
    tts_router.stop_stream()

    print("\n✅ 所有測試完成。")
