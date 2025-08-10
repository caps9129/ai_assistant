import threading
import queue
import time
import uuid
import itertools

from audio.perception.pipeline import PerceptionPipeline
from audio.processor import VoiceProcessor
from transcribers.router import AsrRouter
from tts.router import LanguageRouterTTS
from memory.manager import MemoryManager
from agents.manager import AgentManager
from utils.sound_player import SoundPlayer  # 系統提示音播放器

wakeword_event = threading.Event()
asr_result_queue = queue.Queue()


def handle_wakeword():
    """喚醒詞回調"""
    if not wakeword_event.is_set():
        wakeword_event.set()


def handle_asr_result(result):
    """ASR 回傳的結果，併入時間戳方便量測延遲"""
    asr_end_time = time.time()
    asr_result_queue.put((result, asr_end_time))


def _safe_last_activity_ts(perception_pipeline) -> float:
    """容錯：若 pipeline 沒有 last_activity_ts()，用當前時間替代"""
    try:
        return float(perception_pipeline.last_activity_ts())
    except Exception:
        return time.time()


def main():
    print("🚀 Initializing AI Assistant (Core Components)...")

    global processor
    asr_router = AsrRouter(enable_openai=True)
    agent_manager = AgentManager()

    processor = VoiceProcessor(on_wakeword=handle_wakeword)
    processor.start()

    try:
        while True:
            # ==================================================================
            # --- IDLE STATE ---
            # ==================================================================
            print("\n--- State: IDLE (Waiting for wake word) ---")
            wakeword_event.clear()
            processor.set_state("IDLE")

            wakeword_event.wait()
            print("👋 Wake word detected! Starting new dialog session...")

            # ==================================================================
            # --- DIALOG SESSION SETUP ---
            # ==================================================================
            print("[Main] Initializing services for the new dialog session...")

            # 每次對話建立全新 TTS（避免舊狀態影響）
            tts_router = LanguageRouterTTS(tts_provider="openai")
            sys_tts_router = LanguageRouterTTS(tts_provider="openai-sys")

            # Perception pipeline（含 N/M 平滑 + 延遲 duck + commit）
            perception_pipeline = PerceptionPipeline(
                asr_router=asr_router,
                on_result_callback=handle_asr_result,
                on_speech_onset=lambda: tts_router.duck(-15),
                on_speech_commit=lambda: tts_router.pause_output(flush=False),
                on_speech_cancel=tts_router.unduck,
                on_stop_current_utterance=tts_router.stop_current_utterance,
                on_resume_continue=lambda: tts_router.resume_output(
                    flush=False),
                on_resume_flush=lambda: tts_router.resume_output(flush=True),
                commit_min_ms=360,        # 300–450ms 看體感
                onset_duck_delay_ms=150,  # 120–180ms 看體感
                window_frames=12,         # M
                window_min_speech=9,      # N
                vad_frame_ms=10,          # 10ms 更靈敏
            )
            processor.perception_pipeline = perception_pipeline

            # 系統提示音：等第一個 token 時循環播放
            sys_sound_player = SoundPlayer("utils/sys_sound.wav")

            # 啟動背景服務
            perception_pipeline.start()
            if hasattr(tts_router, 'start_stream'):
                tts_router.start_stream()
            if hasattr(sys_tts_router, 'start_stream'):
                sys_tts_router.start_stream()

            SSID = uuid.uuid4()
            current_memory_session = MemoryManager(session_id=SSID)
            processor.set_state("DIALOG")

            # --- DIALOG FSM ---
            current_dialog_state = "GREETING"
            user_text = None
            response_generator = None
            asr_end_time = None  # 記錄最後一次 ASR 完成時間

            while current_dialog_state != "EXIT":
                print(
                    f"\n[FSM TRACE] Top of loop. Current state: {current_dialog_state}")

                if current_dialog_state == "GREETING":
                    print("--- Dialog State: GREETING ---")
                    # 開場用短提示音，避免 TTS 冷啟延遲
                    greet_player = SoundPlayer("utils/sys_sound.wav")
                    greet_player.start()
                    time.sleep(1.0)
                    greet_player.stop()
                    current_dialog_state = "LISTENING"

                elif current_dialog_state == "LISTENING":
                    print("--- Dialog State: LISTENING (Waiting for command) ---")
                    perception_pipeline.resume()

                    LISTEN_TIMEOUT = 30.0  # 秒
                    deadline = time.time() + LISTEN_TIMEOUT
                    last_seen_activity = _safe_last_activity_ts(
                        perception_pipeline)

                    user_text = ""
                    asr_end_time = None

                    while True:
                        remaining = max(0.0, deadline - time.time())
                        poll = min(0.2, remaining) if remaining > 0 else 0.0
                        try:
                            asr_result, asr_end_time = asr_result_queue.get(
                                timeout=poll)
                            txt = (asr_result or {}).get("text", "").strip()
                            if txt:
                                user_text = txt
                                current_dialog_state = "PROCESSING"
                                break
                            else:
                                continue
                        except queue.Empty:
                            pass

                        cur_activity = _safe_last_activity_ts(
                            perception_pipeline)
                        if cur_activity > last_seen_activity:
                            deadline = time.time() + LISTEN_TIMEOUT
                            last_seen_activity = cur_activity

                        if time.time() >= deadline:
                            print(
                                "⏰ Dialog timed out. Checking for last-second commands...")
                            perception_pipeline.pause()
                            try:
                                final_asr_result, asr_end_time = asr_result_queue.get(
                                    block=False)
                                print(
                                    "🏃‍➡️ Caught a last-second command! Processing it...")
                                user_text = (final_asr_result or {}).get(
                                    "text", "").strip()
                                if user_text:
                                    current_dialog_state = "PROCESSING"
                                else:
                                    sys_tts_router.say("See you next time")
                                    current_dialog_state = "EXIT"
                            except queue.Empty:
                                print("No last-second command. Exiting dialog.")
                                sys_tts_router.say("See you next time")
                                current_dialog_state = "EXIT"
                            break  # 跳出 LISTENING while

                elif current_dialog_state == "PROCESSING":
                    print(
                        f"--- Dialog State: PROCESSING (User said: '{user_text}') ---")
                    # 等 LLM 期間不收音
                    perception_pipeline.pause()

                    # 播放系統提示音直到第一個 token
                    sys_sound_player.start()

                    agent_start_time = time.time()
                    current_memory_session.add_message("user", user_text)
                    response_generator = agent_manager.stream_request(
                        user_text=user_text, memory=current_memory_session
                    )

                    try:
                        # 阻塞拿到第一個 token
                        first_chunk = next(response_generator)
                        # 一拿到就停提示音
                        sys_sound_player.stop()

                        agent_first_token_time = time.time()
                        cognition_latency = (
                            agent_first_token_time - agent_start_time) * 1000
                        print(
                            f"LATENCY_METRIC: Cognition (Agent start -> First token) = {cognition_latency:.2f} ms")

                        if asr_end_time:
                            e2e_latency = (
                                agent_first_token_time - asr_end_time) * 1000
                            print(
                                f"LATENCY_METRIC: ASR End -> First token = {e2e_latency:.2f} ms")

                        if first_chunk == "[EXIT_DIALOG]":
                            print("[Main] User requested to exit. Ending dialog.")
                            current_dialog_state = "EXIT"
                            continue

                        response_generator = itertools.chain(
                            [first_chunk], response_generator)
                        current_dialog_state = "RESPONDING"

                    except StopIteration:
                        sys_sound_player.stop()
                        print(
                            "[Main] Agent returned an empty response. Returning to listening.")
                        current_dialog_state = "LISTENING"
                    except Exception:
                        sys_sound_player.stop()
                        raise

                elif current_dialog_state == "RESPONDING":
                    print("--- Dialog State: RESPONDING (AI speaking via stream) ---")

                    # 用路由器提供的封裝 API：開啟串流（回傳 thread + cancel_event）
                    tts_thread, tts_cancel_event = tts_router.begin_stream(
                        response_generator)

                    # 播放同時恢復收音，等待可能的 barge-in
                    perception_pipeline.resume()

                    interrupted = False
                    while True:
                        # 1) 有沒有 ASR 結果（插話）
                        try:
                            interruption_result, _ = asr_result_queue.get(
                                block=False)
                            interruption_text = (interruption_result or {}).get(
                                'text', '').strip()
                            if interruption_text:
                                print(
                                    f"⚡️ Valid Interruption: '{interruption_text}'.")
                                user_text = interruption_text
                                interrupted = True

                                # 原子取消舊流（封裝了 cancel -> stop_current -> join -> flush）
                                tts_router.cancel_stream_and_reset(
                                    tts_thread, tts_cancel_event)
                                break
                        except queue.Empty:
                            pass

                        # 2) 若 TTS 已播完且 pipeline 沒在處理（沒有 barge-in），即可退出
                        if (not tts_thread.is_alive()) and (not perception_pipeline.is_processing):
                            break

                        time.sleep(0.05)

                    if interrupted:
                        current_dialog_state = "PROCESSING"
                    else:
                        perception_pipeline.pause()
                        current_dialog_state = "LISTENING"

            # -- End of Dialog Loop --
            print("...Exiting dialog loop, cleaning up session services.")

            # 徹底停止和清理服務
            if 'tts_router' in locals() and hasattr(tts_router, 'stop_stream'):
                tts_router.stop_stream()
            if 'sys_tts_router' in locals() and hasattr(sys_tts_router, 'stop_stream'):
                sys_tts_router.stop_stream()
            if 'perception_pipeline' in locals() and perception_pipeline:
                perception_pipeline.stop()

            # 保險：確保系統音停止
            if 'sys_sound_player' in locals() and sys_sound_player:
                sys_sound_player.stop()

            processor.perception_pipeline = None
            while not asr_result_queue.empty():
                asr_result_queue.get_nowait()
            current_memory_session = None

    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    finally:
        print("Gracefully stopping all services...")
        if 'processor' in locals() and processor:
            processor.stop()
        print("All services stopped. Goodbye.")


if __name__ == '__main__':
    processor = None
    main()
