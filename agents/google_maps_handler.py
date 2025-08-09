import json
from openai import OpenAI


# Imports from your project structure
from prompts.manager import PROMPT_MANAGER
from agent_config import AGENT_REGISTRY
from utils.location import LocationManager
from .tools.google_maps_tool import GoogleMapsTool, Maps_TOOLS_DEFINITION


class GoogleMapsHandler:
    def __init__(self):
        config = AGENT_REGISTRY.get_config("GoogleMapsHandler")
        conclusion_config = AGENT_REGISTRY.get_config(
            "GoogleMapsConclusionHandler")

        if not config:
            raise ValueError(
                "Config for 'GoogleMapsHandler' not found in registry.")

        if not conclusion_config:
            raise ValueError(
                "Config for 'GoogleMapsHandler' not found in registry.")

        self.config = config
        self.conclusion_config = conclusion_config
        self.llm_client = OpenAI()

        # The handler creates and owns its tool and other utilities
        self.tool = GoogleMapsTool()
        self.location_manager = LocationManager()

        # The JSON schema description of our tools, imported from the tool file
        self.tools_description = Maps_TOOLS_DEFINITION

        # This dispatcher maps the tool name (string) to the actual Python function
        self.available_tools = {
            "find_nearby_places": self.tool.find_nearby_places,
            "get_directions": self.tool.get_directions,
            "get_place_details": self.tool.get_place_details,
            "geocode_address": self.tool.geocode_address
        }

        print("✅ Google Maps Handler with Tool Use ready.")

    def process(self, user_text: str) -> str:
        """
        Orchestrates the full "Tool Use" flow.
        This method is now self-contained and does not require the memory object.
        """

        system_prompt = PROMPT_MANAGER.get(self.config.prompt_name)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ]

        # --- 3. LLM Call 1: Decide which tool to use ---
        print("-> Step 1: Asking LLM which tool to use...")
        response = self.llm_client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            tools=self.tools_description,
            tool_choice="auto"
        )
        response_message = response.choices[0].message

        if not response_message.tool_calls:
            return response_message.content or "I'm not sure how to help with that map request."

        # --- 4. Execute the chosen tool ---
        tool_messages = [response_message]
        # messages.append(response_message)

        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            tool_function = self.available_tools.get(function_name)

            if not tool_function:
                continue

            function_args = json.loads(tool_call.function.arguments)
            print(
                f"--> LLM decided to call tool '{function_name}' with args: {function_args}")
            tool_output = tool_function(**function_args)

            tool_messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": json.dumps(tool_output, ensure_ascii=False),
            })

        # --- 5. LLM Call 2: Summarize the result ---
        print("-> Step 2: Sending tool result to LLM for final response...")

        system_prompt = PROMPT_MANAGER.get(self.conclusion_config.prompt_name)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ]
        messages.extend(tool_messages)
        final_response = self.llm_client.chat.completions.create(
            model=self.conclusion_config.model,
            messages=messages,
        )

        return final_response.choices[0].message.content


# --- Standalone Test Block ---
if __name__ == '__main__':
    # To run this test, execute from your project root:
    # python -m agents.Maps_handler

    import os
    import sys

    # This allows the script to find root-level modules like 'config' and 'utils'
    project_root = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(project_root)

    print("\n" + "="*50)
    print("Running tests for GoogleMapsHandler...")
    print("="*50 + "\n")

    # Initialize the handler
    handler = GoogleMapsHandler()

    # A comprehensive list of 5 test cases to check all tool functions
    test_cases = [
        # --- Your Original 5 Cases (They are good!) ---
        # 1. Tests find_nearby_places with ranking and a high rating count
        "what's the best rated coffee shop near me and the rating number should be higher than 500",

        # 2. Tests get_directions (and the improved prompt for default origin)
        "how long does it take to drive to the Portland airport",

        # 3. Tests get_place_details for operating hours
        "Is the Safeway on Philomath Blvd still open right now?",

        # 4. Tests get_place_details for a different field (phone number)
        "What is the phone number for the Corvallis Public Library?",

        # 5. Tests geocode_address
        "Where exactly is Oregon State University?",

        # --- NEW: 5 Additional Cases for Broader Coverage ---

        # 6. Tests the new `open_now` and `max_price_level` filters
        "find me a cheap restaurant nearby that is open now",

        # 7. Tests a different travel mode for directions
        "how do I walk to Central Park from here?",

        # 8. Tests a more complex `find_nearby_places` query with multiple constraints
        "I want to find a highly-rated park with at least 50 reviews",

        # 9. Tests a specific `get_place_details` query that requires the LLM to infer the field
        "what time does the downtown Starbucks close today?",

        # 10. Tests an ambiguous query that should trigger a clarification from the LLM
        "I need directions"  # Should fail gracefully or ask "Directions to where?"
    ]

    for text in test_cases:
        print(f"--- Processing User Input: '{text}' ---")
        # The memory parameter is no longer needed
        response = handler.process(text)
        print(f"==> Assistant's Final Response: '{response}'\n")
