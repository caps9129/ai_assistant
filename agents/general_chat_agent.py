from openai import OpenAI
from agent_config import AGENT_REGISTRY
from prompts.manager import PROMPT_MANAGER
from memory.manager import MemoryManager


class GeneralChatAgent:
    def __init__(self):
        print("Initializing GeneralChatAgent...")
        self.client = OpenAI()
        self.config = AGENT_REGISTRY.get_config("GeneralChatAgent")
        if not self.config:
            raise ValueError("Config for 'GeneralChatAgent' not found.")
        # [MODIFIED] 確保從 config 讀取 prompt 名稱
        self.prompt = PROMPT_MANAGER.get(self.config.prompt_name)
        print("✅ GeneralChatAgent is ready.")

    def process_request(self, user_text: str, memory: MemoryManager) -> str:
        """
        處理請求並返回一個完整的字串回應。
        (此方法完全按照您提供的程式碼進行了更新)
        """
        print("-> Delegated to GeneralChatAgent for conversational response.")

        # 從 PromptManager 取得完整、已格式化的系統提示
        system_prompt = PROMPT_MANAGER.get(self.config.prompt_name)

        # 完整的上下文包含：系統提示 (AI 的人設)、歷史對話、以及使用者最新的問題
        messages = [
            {"role": "system", "content": system_prompt},
            *memory.get_history(),  # 這是實現多輪對話的關鍵
            {"role": "user", "content": user_text}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ Error during General Chat LLM call: {e}")
            return "抱歉，我現在有點問題，沒辦法回答您。"

    # [NEW] 新增的、返回文字流 (Generator) 的方法
    def stream_request(self, user_text: str, memory: MemoryManager):
        """
        以串流方式處理請求，並 yield 回應的文字片段。
        """
        print("-> Delegated to GeneralChatAgent for conversational streaming response.")

        # 使用與 process_request 完全相同的邏輯來建構 messages
        system_prompt = PROMPT_MANAGER.get(self.config.prompt_name)
        messages = [
            {"role": "system", "content": system_prompt},
            *memory.get_history(),
            {"role": "user", "content": user_text}
        ]

        try:
            stream = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                stream=True  # <--- 啟用串流模式
            )

            # 迭代 API 回傳的串流，並即時產出內容
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as e:
            print(f"❌ Error during General Chat LLM stream call: {e}")
            # 在串流模式下，我們也 yield 一個錯誤訊息
            yield "抱歉，我現在有點問題，沒辦法回答您。"
