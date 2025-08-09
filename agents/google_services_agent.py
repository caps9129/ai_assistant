import json
from openai import OpenAI

# Import the central registries and handlers
from agent_config import AGENT_REGISTRY
from prompts.manager import PROMPT_MANAGER
from .google_calendar_tasks_handler import GoogleCalendarTasksHandler
from .google_maps_handler import GoogleMapsHandler


class GoogleServicesAgent:
    def __init__(self):
        print("Initializing Google Services Agent Coordinator...")

        # Load the coordinator's own config for routing
        self.router_config = AGENT_REGISTRY.get_config(
            "GoogleServicesAgent_Router")
        if not self.router_config:
            raise ValueError(
                "Config for 'GoogleServicesAgent_Router' not found in registry.")

        # Create instances of the specialized handlers
        self.calendar_tasks_handler = GoogleCalendarTasksHandler()
        self.maps_handler = GoogleMapsHandler()
        self.llm_client = OpenAI()

        # The dispatcher map that connects intents to handler objects
        self.dispatcher = {
            "CALENDAR_TASKS": self.calendar_tasks_handler,
            "MAPS": self.maps_handler,
        }
        print("✅ Google Services Agent is ready.")

    def _get_service_intent(self, user_text: str) -> str:
        """[LLM Call 1] A fast, cheap call to determine which sub-service is needed."""
        print("-> Step 1: Determining which Google service is needed...")

        # --- FIXED: Use the PromptManager to get the rendered prompt ---
        # Get the prompt NAME from the config, then get the full prompt from the manager
        system_prompt = PROMPT_MANAGER.get(self.router_config.prompt_name)

        response = self.llm_client.chat.completions.create(
            model=self.router_config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ],
            temperature=0
        )
        service_intent = response.choices[0].message.content.strip().upper()
        print(f"--> Service determined: {service_intent}")
        return service_intent

    def process_request(self, user_text: str) -> str | None:
        """
        Orchestrates the request by first routing to a sub-service,
        then delegating to the appropriate specialist handler.
        """
        service_intent = self._get_service_intent(user_text)
        handler = self.dispatcher.get(service_intent)

        if not handler:
            print(
                f"--> Intent '{service_intent}' is not handled by this agent.")
            return None

        # Delegate the request to the handler's process method
        return handler.process(user_text)


# ==============================================================================
# == STANDALONE TEST BLOCK
# ==============================================================================
if __name__ == '__main__':
    # To run this test, execute from your project root:
    # python -m agents.google_services_agent

    import os
    import sys

    # This allows the script to find root-level modules like 'agent_config'
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(project_root)

    # Import the agent class after setting up the path

    print("\n" + "="*50)
    print("Running tests for the hierarchical GoogleServicesAgent...")
    print("="*50 + "\n")

    agent = GoogleServicesAgent()

    test_cases = [
        # "remind me to buy milk",
        # "schedule a team meeting tomorrow at 10am",
        "can you schedule a meeting with Jane at 4 PM to finalize the report and also add a task to send her the agenda",
        # "where is the nearest coffee shop",
        # "幫我找一下附近評價比較好的中餐廳，評論人數要大於200，謝謝您",
        # "tell me a fun fact about the ocean",
        # "Is downtown Starbucks open now?",
    ]

    for text in test_cases:
        print(f"--- Processing User Input: '{text}' ---")
        # The call no longer needs a memory object
        response = agent.process_request(text)

        if response is None:
            print(
                "==> Assistant's Final Response: (Agent chose not to handle, as expected for UNKNOWN intent)")
        else:
            print(f"==> Assistant's Final Response: '{response}'\n")
