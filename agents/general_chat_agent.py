# agents/general_chat_agent.py
from __future__ import annotations
from agent_config import AGENT_REGISTRY
from agents.base import BaseAgentConfig, OpenAIChatAgentBase


class GeneralChatAgent(OpenAIChatAgentBase):
    def __init__(self):
        cfg = AGENT_REGISTRY.get_config("GeneralChatAgent")
        if not cfg:
            raise ValueError("Config for 'GeneralChatAgent' not found.")

        base_cfg = BaseAgentConfig(
            id="GeneralChatAgent",
            prompt_name=cfg.prompt_name,    # e.g. "general_chat_agent"
            model=cfg.model,
            temperature=getattr(cfg, "temperature", 0.0),
            max_tokens=getattr(cfg, "max_tokens", None),
            history_limit_pairs=getattr(cfg, "history_limit_pairs", 10),
            max_chars=getattr(cfg, "max_chars", 12000),
            output_mode="text",
        )
        super().__init__(base_cfg)
        print("✅ GeneralChatAgent ready.")


if __name__ == '__main__':

    from agents.general_chat_agent import GeneralChatAgent
    from memory.manager import MemoryManager

    agent = GeneralChatAgent()
    memory = MemoryManager()

    user_text = "跟我說個輕鬆的小笑話"
    for chunk in agent.stream_request(user_text, memory):
        print(chunk, end="", flush=True)
    print()
