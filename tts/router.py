# tts/router.py

from openai import OpenAI
import threading
import time
from pathlib import Path
import fasttext
import nltk
import re
from typing import Optional, Tuple, Iterable

# 導入我們設計的設定檔和 TTS 引擎
from .config import TTS_CONFIG
from .base_engine import TTSEngineBase
from .edge_tts_engine import EdgeTTSEngine
from .openai_tts_engine import OpenAITTSEngine

# 確保 NLTK 的斷句模型存在
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
            return "en", 0.9
        predictions = self.model.predict(text, k=1)
        lang_code = predictions[0][0].replace('__label__', '')
        confidence = predictions[1][0]
        return lang_code, confidence


class LanguageRouterTTS:
    # ---- A：合併策略預設值（可視需求微調） ----
    COALESCE_MIN_CHARS = 160      # 累積到這麼多字就出聲
    COALESCE_MAX_SENT = 3         # 一個 block 最多幾句
    COALESCE_MAX_CHARS = 600      # 安全上限（避免一次呼叫過長）

    # 條列點偵測（數字/符號/中英混合清單）
    LIST_HEAD_RE = re.compile(
        r'^\s*(?:[\u2022•\-*]|[0-9]{1,2}[.)]|[一二三四五六七八九十]{1,3}[、.])\s+'
    )

    def __init__(self, stop_event: Optional[threading.Event] = None, tts_provider: str = None):
        print("Initializing LanguageRouterTTS...")

        self.stop_event = stop_event if stop_event else threading.Event()
        self.detector = FastTextLangDetector()

        self.config = TTS_CONFIG
        self.active_provider = tts_provider or self.config.get(
            "active_provider", "edge")
        print(f"... Active TTS provider: {self.active_provider}")

        self.tts_engines: dict[str, TTSEngineBase] = {}
        self._initialize_engines()

        # 追蹤目前的串流（方便外層不用自己記）
        self._current_stream_thread: Optional[threading.Thread] = None
        self._current_cancel_event: Optional[threading.Event] = None
        self._stream_lock = threading.Lock()  # 保護上面兩個欄位

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
                # 大 backlog + 積極吸收，避免 pause 期間背壓
                backlog_frames_limit=12000,
                pause_drain_batch=24,
                # 如需更平順可在這裡加上 prebuffer_min_frames=3（OpenAITTSEngine 支援）
                # prebuffer_min_frames=3,
            )

    # ---------------- Lifecycle ---------------- #
    def start_stream(self):
        print("[TTS Router] Starting persistent streams for all engines...")
        for engine in self.tts_engines.values():
            if hasattr(engine, 'start_stream'):
                engine.start_stream()

    def stop_stream(self):
        print("[TTS Router] Stopping persistent streams for all engines...")
        for engine in self.tts_engines.values():
            if hasattr(engine, 'stop_stream'):
                engine.stop_stream()

    # ---------------- Core Speak APIs ---------------- #
    def _pick_engine(self, text: str) -> Optional[TTSEngineBase]:
        if self.active_provider == "openai":
            return self.tts_engines.get("en")
        else:
            lang_code, _ = self.detector.detect(text)
            return self.tts_engines.get(lang_code, self.tts_engines.get("en"))

    def say(self, text: str):
        if not text or not text.strip():
            return
        engine = self._pick_engine(text)
        if engine:
            engine.say(text)
        else:
            print(
                f"ERROR: Could not find a suitable TTS engine for provider '{self.active_provider}'")

    # ---- helpers for A ----
    @staticmethod
    def _clean_piece(s: str) -> str:
        # 壓空白，但保留換行（清單時好聽一點）
        return re.sub(r'[ \t]+', ' ', s).strip()

    @classmethod
    def _is_list_head(cls, s: str) -> bool:
        return bool(cls.LIST_HEAD_RE.match(s))

    @staticmethod
    def _join_block(parts: list[str]) -> str:
        # 若包含條列，改用換行；否則用空格連
        if any(LanguageRouterTTS._is_list_head(p) for p in parts):
            return "\n".join(parts)
        return " ".join(parts)

    def say_stream(self, text_generator: Iterable[str], cancel_event: Optional[threading.Event] = None):
        """
        A：句子合併的串流播放器
        - 把多個句子/條列合成一個 block 再送給 TTS，降低句間握手延遲
        - 不做 sleep 型 hold，靠「長度/句數門檻」達成自然的合併
        - 支援 cancel_event，一旦 set() 會盡快中止；若同時呼叫 stop_current_utterance()，可立即打斷目前播放
        """
        print("\n[TTS Router] Starting to process text stream with NLTK...")
        remainder = ""                # 尚未完整成句的尾巴
        block_parts: list[str] = []   # 累積要一次播的句子
        block_len = 0

        def _flush_block():
            nonlocal block_parts, block_len
            if not block_parts:
                return
            # 再次檢查是否已取消（降低競態）
            if cancel_event is not None and cancel_event.is_set():
                print("[TTS Router] say_stream canceled before flush.")
                block_parts.clear()
                block_len = 0
                return
            block = self._join_block(block_parts)
            print(
                f"[TTS Router] Speaking block ({len(block_parts)} sent, {len(block)} chars)")
            self.say(block)
            block_parts.clear()
            block_len = 0

        for text_chunk in text_generator:
            if cancel_event is not None and cancel_event.is_set():
                print("[TTS Router] say_stream canceled.")
                break
            if self.stop_event.is_set():
                print("[TTS Router] Stop event detected, stopping stream playback.")
                break

            remainder += text_chunk
            # 以 NLTK 切句；最後一個視為尚未完成的 remainder
            sentences = nltk.sent_tokenize(remainder)

            if len(sentences) > 1:
                completed = sentences[:-1]
                remainder = sentences[-1]

                for s in completed:
                    piece = self._clean_piece(s)
                    if not piece:
                        continue
                    block_parts.append(piece)
                    block_len += len(piece)

                    # 合併條件：達到最小長度，或句數到頂，或安全上限
                    if (
                        block_len >= self.COALESCE_MIN_CHARS
                        or len(block_parts) >= self.COALESCE_MAX_SENT
                        or block_len >= self.COALESCE_MAX_CHARS
                    ):
                        _flush_block()
                        if cancel_event is not None and cancel_event.is_set():
                            break

        # 讀流結束後，處理尾巴
        if not (cancel_event and cancel_event.is_set()) and not self.stop_event.is_set():
            tail = self._clean_piece(remainder)
            if tail:
                block_parts.append(tail)
                block_len += len(tail)
            _flush_block()

        print("[TTS Router] Text stream finished.")

    # --------- 便利方法：開始/取消 串流回覆（原子取消用） ---------
    def begin_stream(self, text_generator: Iterable[str]) -> Tuple[threading.Thread, threading.Event]:
        """
        啟動一個 say_stream 執行緒並回傳 (thread, cancel_event)。
        也會把這對物件記錄在 router 內部，方便之後 cancel_stream_and_reset() 不帶參數直接用。
        """
        cancel_event = threading.Event()
        t = threading.Thread(target=self.say_stream, args=(
            text_generator, cancel_event), daemon=True)
        with self._stream_lock:
            self._current_stream_thread = t
            self._current_cancel_event = cancel_event
        t.start()
        return t, cancel_event

    def cancel_stream_and_reset(self, tts_thread: Optional[threading.Thread] = None,
                                cancel_event: Optional[threading.Event] = None,
                                join_timeout: float = 0.5):
        """
        原子取消目前的 TTS 串流並清掉所有殘留，避免「上一段漏播」：
        1) 讓 say_stream 立刻跳出（cancel_event.set()）
        2) 停掉當前 producer + 丟棄 callback 當前片段（stop_current_utterance）
        3) 等 say_stream 執行緒收尾（join）
        4) 清空 backlog/queue（resume_output(flush=True)）
        """
        # 使用傳參或已記錄的那一組
        with self._stream_lock:
            t = tts_thread or self._current_stream_thread
            ev = cancel_event or self._current_cancel_event

        if ev is not None:
            ev.set()

        try:
            self.stop_current_utterance()
        except Exception:
            pass

        if t is not None and t.is_alive():
            t.join(timeout=join_timeout)

        try:
            # 丟棄任何殘留的 backlog/queue，確保「乾淨起頭」
            self.resume_output(flush=True)
        except Exception:
            pass

        # 清掉記錄
        with self._stream_lock:
            if t is None or (self._current_stream_thread is t):
                self._current_stream_thread = None
            if ev is None or (self._current_cancel_event is ev):
                self._current_cancel_event = None

    # ---------------- Barge-in Helpers ---------------- #
    def duck(self, db: float = -18.0):
        for engine in self.tts_engines.values():
            if hasattr(engine, "duck"):
                engine.duck(db)

    def unduck(self):
        for engine in self.tts_engines.values():
            if hasattr(engine, "unduck"):
                engine.unduck()

    def stop_current_utterance(self):
        for engine in self.tts_engines.values():
            if hasattr(engine, "stop_current_utterance"):
                engine.stop_current_utterance()

    # === 暫停/恢復輸出 ===
    def pause_output(self, *, flush: bool = False, timeout_ms: int | None = None):
        for engine in self.tts_engines.values():
            if hasattr(engine, "pause_output"):
                try:
                    engine.pause_output(flush=flush, timeout_ms=timeout_ms)
                except TypeError:
                    engine.pause_output(flush=flush)

    def resume_output(self, *, flush: bool = False):
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
        messages=[{"role": "user", "content": "介紹一下何謂太陽系，列點說明並補充兩三句話。"}],
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
    print(">>> 測試: 串流 say_stream() 方法 (with coalescing + atomic cancel)")
    print("="*40)

    tts_router.start_stream()
    llm_generator = simulate_llm_stream()

    # 用 begin_stream 啟動，並保留 (thread, cancel_event)
    tts_thread, cancel_ev = tts_router.begin_stream(llm_generator)
    print("TTS 播放執行緒已在背景啟動...")

    # 播放幾秒後，模擬插話 → 原子取消
    time.sleep(5)
    print(">>> 模擬插話：cancel_stream_and_reset()")
    tts_router.cancel_stream_and_reset(tts_thread, cancel_ev)

    # 再播另一段以確保乾淨起頭（此處直接呼叫 say）
    tts_router.say("好的，已經切換到新的回覆段落。")

    # 等待執行緒收尾
    if tts_thread.is_alive():
        tts_thread.join()
    print("\n✅ 測試完成。")
