# tts/router.py

from openai import OpenAI
import threading
import time
from pathlib import Path
import fasttext
import nltk

# 導入我們設計的設定檔和 TTS 引擎
from .config import TTS_CONFIG
from .base_engine import TTSEngineBase
from .edge_tts_engine import EdgeTTSEngine
from .openai_tts_engine import OpenAITTSEngine

# 確保 NLTK 的斷句模型存在，如果不存在則自動下載
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

        for lang, cfg in provider_config.items():
            print(
                f"... Initializing '{lang}' voice using {self.active_provider}...")
            self.tts_engines[lang] = EngineClass(
                voice=cfg.voice,
                stop_event=self.stop_event,
                model=cfg.model,
                sample_rate=cfg.sample_rate,
                backlog_frames_limit=12000,
                pause_drain_batch=24
            )

    # ---------------- Lifecycle ---------------- #
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

    # ---------------- Core Speak APIs ---------------- #
    def _pick_engine(self, text: str) -> TTSEngineBase | None:
        """依語言/供應商挑選一個合適的引擎"""
        if self.active_provider == "openai":
            # OpenAI 模型本身不分語言，固定用 'en' 這個 slot
            return self.tts_engines.get("en")
        else:
            lang_code, _ = self.detector.detect(text)
            return self.tts_engines.get(lang_code, self.tts_engines.get("en"))

    def say(self, text: str):
        """播放單一、完整的文字字串。"""
        if not text or not text.strip():
            return
        engine = self._pick_engine(text)
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

    # ---------------- Barge-in Helpers ---------------- #
    def duck(self, db: float = -18.0):
        """將所有引擎音量降低（duck）。"""
        for engine in self.tts_engines.values():
            if hasattr(engine, "duck"):
                engine.duck(db)

    def unduck(self):
        """將所有引擎音量恢復。"""
        for engine in self.tts_engines.values():
            if hasattr(engine, "unduck"):
                engine.unduck()

    def stop_current_utterance(self):
        """輕量中斷目前這句（不關閉底層 stream）。"""
        for engine in self.tts_engines.values():
            if hasattr(engine, "stop_current_utterance"):
                engine.stop_current_utterance()

    # === NEW: 暫停/恢復輸出（不關閉串流、不中斷 producer） ===
    def pause_output(self, *, flush: bool = False, timeout_ms: int | None = None):
        """
        暫停所有引擎的『播放端』：
        - flush=False：將後續產生的音訊累積到各引擎的 backlog，之後可接續播放
        - flush=True：清掉 backlog（保留串流/連線）
        - timeout_ms：可選。若引擎支援，會覆寫暫停逾時（逾時自動 resume+flush）
        """
        for engine in self.tts_engines.values():
            if hasattr(engine, "pause_output"):
                try:
                    engine.pause_output(flush=flush, timeout_ms=timeout_ms)
                except TypeError:
                    # 舊版引擎只接受 flush 參數
                    engine.pause_output(flush=flush)

    def resume_output(self, *, flush: bool = False):
        """
        恢復所有引擎的『播放端』：
        - flush=False：把暫停期間累積的 backlog 先播完再繼續新內容
        - flush=True：丟棄 backlog，直接從最新內容繼續
        """
        for engine in self.tts_engines.values():
            if hasattr(engine, "resume_output"):
                engine.resume_output(flush=flush)

    # ---------------- Misc ---------------- #
    def stop(self):
        self.stop_event.set()

    def reset(self):
        print("[TTS Router] Resetting engines...")
        self.stop_stream()
        self._initialize_engines()
        self.start_stream()


# --- 測試方法 ---

def simulate_llm_stream():
    _client = OpenAI()
    stream = _client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "介紹一下何謂太陽系"}],
        stream=True,
    )
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content


if __name__ == '__main__':
    PROVIDER_TO_TEST = "openai"
    tts_router = LanguageRouterTTS(tts_provider=PROVIDER_TO_TEST)

    print("\n" + "="*40)
    print(">>> 測試 3: 串流 say_stream() 方法 (with router duck/unduck)")
    print("="*40)

    tts_router.start_stream()
    llm_generator = simulate_llm_stream()
    tts_thread = threading.Thread(
        target=tts_router.say_stream, args=(llm_generator,))
    tts_thread.start()
    print("TTS 播放執行緒已在背景啟動...")

    print("主執行緒等待幾秒，讓語音播放一會兒...")
    time.sleep(15)

    print(">>> 現在呼叫 tts_router.duck() 來降低音量！")
    tts_router.duck(-18)

    time.sleep(7)

    print(">>> 現在呼叫 tts_router.unduck() 來恢復音量！")
    tts_router.unduck()

    print("主執行緒等待 TTS 播放完畢...")
    tts_thread.join()
    print("TTS 播放執行緒已結束。")

    print("\n✅ 所有測試完成。")
