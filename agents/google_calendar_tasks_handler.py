import json
from openai import OpenAI
from prompts.manager import PROMPT_MANAGER
from agent_config import AGENT_REGISTRY
from utils.location import LocationManager
from agents.tools.google_calendar_tool import GoogleCalendarTool
from agents.tools.google_tasks_tool import GoogleTasksTool


class GoogleCalendarTasksHandler:
    """
    A specialist that uses a detailed prompt to extract parameters for
    Google Calendar and Tasks, then calls the appropriate tool.
    """

    def __init__(self):
        config = AGENT_REGISTRY.get_config("GoogleCalendarTasksHandler")
        if not config:
            raise ValueError(
                "Config for 'GoogleCalendarTasksHandler' not found.")

        self.config = config
        self.llm_client = OpenAI()
        self.location_manager = LocationManager()  # Needs location for time
        self.calendar_tool = GoogleCalendarTool()
        self.tasks_tool = GoogleTasksTool()
        print("✅ Calendar & Tasks Handler ready.")

    def _extract_parameters(self, user_text: str) -> dict:
        """
        [LLM Call 2 - Specialist] Uses this handler's specific prompt
        to extract detailed parameters for an event or task.
        """
        print("-> Step 2: Extracting detailed parameters for Calendar/Tasks...")

        system_prompt = PROMPT_MANAGER.get(self.config.prompt_name)

        # print(f"51: {system_prompt}")

        response = self.llm_client.chat.completions.create(
            model=self.config.model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ]
        )
        return json.loads(response.choices[0].message.content)

    def process(self, user_text: str) -> str:
        """Processes a single calendar or task action."""
        structured_data = self._extract_parameters(user_text)

        if "clarification" in structured_data:
            return structured_data["clarification"]

        actions = structured_data.get("actions", [])
        if not actions:
            return "I understood you wanted help with your schedule, but couldn't determine a specific action."

        tool_outputs = []

        # Execute all tool calls first
        for action in actions:
            intent = action.get("tool_name")
            params = action.get("parameters")

            tool_output = None
            if intent == "CALENDAR_EVENT_CREATE":
                tool_output = self.calendar_tool.create_event(params)
            elif intent == "TASK_CREATE":
                tool_output = self.tasks_tool.create_task(params)

            if tool_output:
                tool_outputs.append(tool_output)

        if not tool_outputs:
            return "I understood you, but I wasn't able to complete an action."

        # --- LLM Call 2: Summarize the results of all actions ---
        print("-> Step 2: Sending tool results to LLM for final response...")

        final_prompt = f"""
        You are a helpful assistant. The user's original request was: '{user_text}'.
        You have successfully executed one or more actions. The results of your actions are in this JSON:
        {json.dumps(tool_outputs)}
        
        Based on these results, provide a single, concise, and friendly confirmation message to the user.
        """

        final_response = self.llm_client.chat.completions.create(
            model=self.config.model,  # Or a faster model like gpt-3.5-turbo
            messages=[
                {"role": "system", "content": "You are a helpful voice assistant."},
                {"role": "user", "content": final_prompt}
            ]
        )
        return final_response.choices[0].message.content
