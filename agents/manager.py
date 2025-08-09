from openai import OpenAI

from agent_config import AGENT_REGISTRY
from prompts.manager import PROMPT_MANAGER
from memory.manager import MemoryManager

from .google_services_agent import GoogleServicesAgent
from .general_chat_agent import GeneralChatAgent


class AgentManager:
    def __init__(self):
        print("Initializing AgentManager...")
        self.router_config = AGENT_REGISTRY.get_config("MainRouter")
        if not self.router_config:
            raise ValueError(
                "Config for 'MainRouter' not found in agent_config.py.")
        self.llm_client = OpenAI()
        self.specialist_agents = {
            "GOOGLE_SERVICES": GoogleServicesAgent(),
        }
        self.general_chat_agent = GeneralChatAgent()
        print("✅ AgentManager is ready.")

    def _get_top_level_intent(self, user_text: str) -> str:

        print("-> Main Router: Classifying user intent...")
        system_prompt = PROMPT_MANAGER.get(self.router_config.prompt_name)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ]
        response = self.llm_client.chat.completions.create(
            model=self.router_config.model, messages=messages, temperature=0
        )
        intent = response.choices[0].message.content.strip().upper()
        print(f"--> Main intent determined: {intent}")
        return intent

    def route_request(self, user_text: str, memory: MemoryManager) -> str:

        intent = self._get_top_level_intent(user_text)
        specialist_agent = self.specialist_agents.get(intent)

        if specialist_agent:
            response = specialist_agent.process_request(user_text)
            if response is not None:
                return response
            else:
                print(
                    f"--> Specialist '{intent}' could not handle. Falling back to GeneralChatAgent.")
                return self.general_chat_agent.process_request(user_text, memory)
        else:
            return self.general_chat_agent.process_request(user_text, memory)

    # [NEW] 新增的 stream_request 方法，與 route_request 並行
    def stream_request(self, user_text: str, memory: MemoryManager):
        """
        [新方法]
        以串流方式路由請求。
        - 如果是工具型 Agent，則執行並一次性 yield 完整結果。
        - 如果是通用聊天 Agent，則 yield from 其文字流。
        """
        intent = self._get_top_level_intent(user_text)
        if intent == "EXIT":
            yield "[EXIT_DIALOG]"
            return
        specialist_agent = self.specialist_agents.get(intent)

        if specialist_agent:

            response = specialist_agent.process_request(user_text)
            if response is not None:
                print(
                    f"--> Specialist '{intent}' handled the request (non-streamed).")
                yield response
                return
            else:
                # 如果專家無法處理，則 fallback 到通用聊天的串流模式
                print(
                    f"--> Specialist '{intent}' could not handle. Falling back to GeneralChatAgent streaming.")
                # 使用 yield from 將 general_chat_agent 的產生器內容直接傳遞出去
                yield from self.general_chat_agent.stream_request(user_text, memory)
        else:
            # 如果意圖是 GENERAL_CHAT 或 UNKNOWN，直接使用通用聊天的串流模式
            yield from self.general_chat_agent.stream_request(user_text, memory)
