# main.py

import threading
import queue
import time
import uuid
import itertools

# 导入所有需要的模组
from audio.perception.pipeline import PerceptionPipeline
from audio.processor import VoiceProcessor
from transcribers.router import AsrRouter
from tts.router import LanguageRouterTTS
from memory.manager import MemoryManager
from agents.manager import AgentManager

# --- 全域伫列与事件 ---
wakeword_event = threading.Event()
asr_result_queue = queue.Queue()


def handle_wakeword():
    """唤醒词回调函式"""
    if not wakeword_event.is_set():
        wakeword_event.set()


def handle_asr_result(result):
    # [MODIFIED] 我們現在也將 ASR 完成的時間戳一起放入佇列
    asr_end_time = time.time()
    asr_result_queue.put((result, asr_end_time))


def main():
    """AI 助理的主函式"""
    print("🚀 Initializing AI Assistant (Core Components)...")

    # --- 核心元件初始化 (这些是无状态的，可以只初始化一次) ---
    global processor
    asr_router = AsrRouter(enable_openai=True)
    agent_manager = AgentManager()

    # 启动最底层的音讯监听器
    # [FIX] 移除 on_sentence，因为其职责已转移
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

            # 为每一次对话建立全新的、干净的服务实例
            tts_router = LanguageRouterTTS(tts_provider="openai")
            sys_tts_router = LanguageRouterTTS(
                tts_provider="openai-sys")  # 假设有 openai-sys 设定
            perception_pipeline = PerceptionPipeline(
                asr_router=asr_router, on_result_callback=handle_asr_result
            )
            processor.perception_pipeline = perception_pipeline  # 绑定新的 pipeline

            # 启动本次对话所需的背景服务
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
                            timeout=12.0)

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
                            final_asr_result = asr_result_queue.get(
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
                    perception_pipeline.pause()
                    agent_start_time = time.time()
                    current_memory_session.add_message("user", user_text)
                    response_generator = agent_manager.stream_request(
                        user_text=user_text, memory=current_memory_session
                    )

                    try:
                        first_chunk = next(response_generator)
                        agent_first_token_time = time.time()
                        cognition_latency = (
                            agent_first_token_time - agent_start_time) * 1000
                        print(
                            f"LATENCY_METRIC: Cognition (Agent start -> First token) = {cognition_latency:.2f} ms")
                        if first_chunk == "[EXIT_DIALOG]":
                            print("[Main] User requested to exit. Ending dialog.")
                            current_dialog_state = "EXIT"
                            continue

                        response_generator = itertools.chain(
                            [first_chunk], response_generator)
                        current_dialog_state = "RESPONDING"
                    except StopIteration:
                        print(
                            "[Main] Agent returned an empty response. Returning to listening.")
                        current_dialog_state = "LISTENING"

                elif current_dialog_state == "RESPONDING":
                    print("--- Dialog State: RESPONDING (AI speaking via stream) ---")

                    # tts_thread = threading.Thread(
                    #     target=tts_router.say_stream, args=(response_generator,), daemon=True)
                    text_buffer = ""
                    for text_chunk in response_generator:
                        text_buffer += text_chunk
                    tts_thread = threading.Thread(
                        target=tts_router.say, args=(text_buffer,), daemon=True)
                    tts_thread.start()

                    perception_pipeline.resume()

                    interrupted = False
                    while tts_thread.is_alive():
                        try:
                            interruption_result, _ = asr_result_queue.get(
                                block=False)
                            interruption_text = interruption_result.get(
                                'text', '').strip()
                            if interruption_text:
                                print(
                                    f"⚡️ Valid Interruption: '{interruption_text}'.")
                                if hasattr(tts_router, 'stop'):
                                    tts_router.stop()

                                # 重置 TTS 路由器以确保下一次播放的稳定性
                                if hasattr(tts_router, 'reset'):
                                    tts_router.reset()

                                user_text = interruption_text
                                interrupted = True
                                break
                        except queue.Empty:
                            time.sleep(0.1)

                    perception_pipeline.pause()

                    if interrupted:
                        current_dialog_state = "PROCESSING"
                    else:
                        current_dialog_state = "LISTENING"

            # -- End of Dialog Loop --
            print("...Exiting dialog loop, cleaning up session services.")

            # 在对话结束时，彻底停止和清理服务
            if 'tts_router' in locals() and hasattr(tts_router, 'stop_stream'):
                tts_router.stop_stream()
            if 'sys_tts_router' in locals() and hasattr(sys_tts_router, 'stop_stream'):
                sys_tts_router.stop_stream()
            if 'perception_pipeline' in locals() and perception_pipeline:
                perception_pipeline.stop()

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
