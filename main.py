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
    """唤醒词回调函式"""
    if not wakeword_event.is_set():
        wakeword_event.set()


def handle_asr_result(result):
    # 併入完成時間，方便算延遲
    asr_end_time = time.time()
    asr_result_queue.put((result, asr_end_time))


def main():
    """AI 助理的主函式"""
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

            # 為每一次對話建立全新的、乾淨的服務實例
            tts_router = LanguageRouterTTS(tts_provider="openai")
            sys_tts_router = LanguageRouterTTS(
                tts_provider="openai-sys")

            # 掛上 duck/commit/cancel/ASR 回調（Step B）
            perception_pipeline = PerceptionPipeline(
                asr_router=asr_router,
                on_result_callback=handle_asr_result,
                on_speech_onset=lambda: tts_router.duck(-15),
                on_speech_commit=lambda: tts_router.pause_output(
                    flush=False),  # 只暫停，資料寫入 backlog
                on_speech_cancel=tts_router.unduck,
                on_stop_current_utterance=tts_router.stop_current_utterance,   # ASR 有效時一起停掉 producer
                on_resume_continue=lambda: tts_router.resume_output(
                    flush=False),
                on_resume_flush=lambda: tts_router.resume_output(flush=True),
                commit_min_ms=360,           # 300–450ms 看體感
                onset_duck_delay_ms=120,     # 120–180ms 看體感
                window_frames=8,             # M
                window_min_speech=6         # N
                # vad_frame_ms=30,           # 需要再細一點才改 10
            )
            processor.perception_pipeline = perception_pipeline  # 綁定新的 pipeline

            # 進度提示音（等待 LLM 第一個 token 時循環播放）
            sys_sound_player = SoundPlayer("utils/sys_sound.wav")

            # 啟動本次對話所需的背景服務
            perception_pipeline.start()
            if hasattr(tts_router, 'start_stream'):
                tts_router.start_stream()
            if hasattr(sys_tts_router, 'start_stream'):
                sys_tts_router.start_stream()

            SSID = uuid.uuid4()
            current_memory_session = MemoryManager(session_id=SSID)
            processor.set_state("DIALOG")

            # --- DIALOG FSM (Finite State Machine) ---
            current_dialog_state = "GREETING"
            user_text = None
            response_generator = None
            asr_end_time = None  # 記錄最後一次 ASR 完成時間

            while current_dialog_state != "EXIT":
                print(
                    f"\n[FSM TRACE] Top of loop. Current state: {current_dialog_state}")

                if current_dialog_state == "GREETING":
                    print("--- Dialog State: GREETING ---")
                    sys_tts_router.say("Hello! I am your AI assistance")
                    current_dialog_state = "LISTENING"

                elif current_dialog_state == "LISTENING":
                    print("--- Dialog State: LISTENING (Waiting for command) ---")
                    perception_pipeline.resume()
                    try:
                        asr_result, asr_end_time = asr_result_queue.get(
                            timeout=30.0)

                        user_text = asr_result.get('text', '').strip()
                        if user_text:
                            current_dialog_state = "PROCESSING"
                        else:
                            continue
                    except queue.Empty:
                        print(
                            "⏰ Dialog timed out. Checking for last-second commands...")
                        perception_pipeline.pause()
                        try:
                            final_asr_result, asr_end_time = asr_result_queue.get(
                                block=False)
                            print(
                                "🏃‍➡️ Caught a last-second command! Processing it...")
                            user_text = final_asr_result.get(
                                'text', '').strip()
                            if user_text:
                                current_dialog_state = "PROCESSING"
                            else:
                                sys_tts_router.say("See you next time")
                                current_dialog_state = "EXIT"
                        except queue.Empty:
                            print("No last-second command. Exiting dialog.")
                            sys_tts_router.say("See you next time")
                            current_dialog_state = "EXIT"

                elif current_dialog_state == "PROCESSING":
                    print(
                        f"--- Dialog State: PROCESSING (User said: '{user_text}') ---")
                    # 等待 LLM 回應期間不需要收音
                    perception_pipeline.pause()

                    # 播放系統提示音直到第一個 token
                    sys_sound_player.start()

                    agent_start_time = time.time()
                    current_memory_session.add_message("user", user_text)
                    response_generator = agent_manager.stream_request(
                        user_text=user_text, memory=current_memory_session
                    )

                    try:
                        # 阻塞拿到第一個 token：這段期間會播放系統音
                        first_chunk = next(response_generator)

                        # ⏹️ 一拿到第一個 token，立刻停系統音
                        sys_sound_player.stop()

                        agent_first_token_time = time.time()
                        cognition_latency = (
                            agent_first_token_time - agent_start_time) * 1000
                        print(
                            f"LATENCY_METRIC: Cognition (Agent start -> First token) = {cognition_latency:.2f} ms")

                        # 可選：ASR 結束到第一個 token 的整體語音互動延遲
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
                        # ⏹️ 保險：就算 agent 沒回也要確保停掉系統音
                        sys_sound_player.stop()
                        print(
                            "[Main] Agent returned an empty response. Returning to listening.")
                        current_dialog_state = "LISTENING"
                    except Exception:
                        # ⏹️ 發生例外也要停
                        sys_sound_player.stop()
                        raise

                elif current_dialog_state == "RESPONDING":
                    print("--- Dialog State: RESPONDING (AI speaking via stream) ---")

                    # 先把 LLM 串流合併成一句（你現有的實作）
                    try:
                        # 確保不是 paused，也清掉任何遺留 backlog
                        tts_router.resume_output(flush=True)
                    except Exception:
                        pass

                    # text_buffer = ""
                    # for text_chunk in response_generator:
                    #     text_buffer += text_chunk

                    tts_thread = threading.Thread(
                        target=tts_router.say_stream, args=(
                            response_generator,), daemon=True
                    )
                    tts_thread.start()

                    # 播放同時恢復收音，等待可能的 barge-in
                    perception_pipeline.resume()

                    interrupted = False
                    # 新邏輯：觀察 ASR 插話（queue）與 pipeline.is_processing
                    while True:
                        # 1) 有沒有 ASR 結果（插話）
                        try:
                            interruption_result, _ = asr_result_queue.get(
                                block=False)
                            interruption_text = interruption_result.get(
                                'text', '').strip()
                            if interruption_text:
                                print(
                                    f"⚡️ Valid Interruption: '{interruption_text}'.")
                                # （此時 pause_output/duck/… 已由 pipeline commit/ASR 回調處理）
                                user_text = interruption_text
                                interrupted = True
                                break
                        except queue.Empty:
                            pass

                        # 2) 若 TTS 已播完且管線沒有在處理（沒有 barge-in），即可退出
                        if (not tts_thread.is_alive()) and (not perception_pipeline.is_processing):
                            break

                        time.sleep(0.05)

                    # 若 barge-in 成功，保持收音直到 PROCESSING 狀態再 pause
                    if interrupted:
                        current_dialog_state = "PROCESSING"
                    else:
                        perception_pipeline.pause()
                        current_dialog_state = "LISTENING"

            # -- End of Dialog Loop --
            print("...Exiting dialog loop, cleaning up session services.")

            # 在對話結束時，徹底停止和清理服務
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
    # 将需要被全域存取的物件初始化为 None
    processor = None
    main()
