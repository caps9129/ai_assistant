from dataclasses import dataclass
from typing import Dict, Optional

# --- This is the key change ---


@dataclass
class AgentConfig:
    """A structured class to hold the configuration for a single AI agent."""
    name: str
    model: str
    prompt_name: str
    temperature: Optional[float] = None


class AgentRegistry:
    """
    A central, singleton registry to hold and provide access to all agent configurations.
    This ensures that there is a single source of truth for agent settings.
    """
    _instance = None
    _configs: Dict[str, AgentConfig] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AgentRegistry, cls).__new__(cls)
            cls._instance._initialize_configs()
        return cls._instance

    def _initialize_configs(self):
        """
        Private method to define and register all agent configurations.
        """
        self._configs = {
            "MainRouterAgent": AgentConfig(
                name="MainRouterAgent",
                model="gpt-4.1",
                prompt_name="main_router",
                temperature=0.0
            ),
            "ComplexPlannerAgent": AgentConfig(
                name="ComplexPlannerAgent",
                model="gpt-4.1",
                prompt_name="complex_planner"
            ),
            "SimplePlannerAgent": AgentConfig(
                name="SimplePlannerAgent",
                model="gpt-4.1",
                prompt_name="simple_planner"
            ),
            # --- Google Services Agents ---
            "GoogleServicesAgent_Router": AgentConfig(
                name="GoogleServicesAgent_Router",
                model="gpt-4o",  # Using a more standard model name
                prompt_name="google/router"
            ),
            "GoogleCalendarTasksHandler": AgentConfig(
                name="GoogleCalendarTasksHandler",
                model="gpt-4.1",
                prompt_name="google/schedule_handler"  # CORRECTED: Fixed typo
            ),
            "GoogleMapsHandler": AgentConfig(
                name="GoogleMapsHandler",
                model="gpt-4o",  # Using a more standard model name
                prompt_name="google/maps_handler"
            ),
            "GoogleMapsConclusionHandler": AgentConfig(
                name="GoogleMapsConclusionHandler",
                model="gpt-4o",  # Using a more standard model name
                prompt_name="google/maps_conclusion_handler"
            ),

            # --- Other Agents ---

            "WeatherAgent": AgentConfig(
                name="WeatherAgent",
                model="gpt-3.5-turbo",
                prompt_name="weather_agent"
            ),
            "GeneralChatAgent": AgentConfig(
                name="GeneralChatAgent",
                model="gpt-4o",
                prompt_name="general_chat_agent",
                temperature=0.7
            )
        }
        print(
            f"✅ AgentRegistry initialized with {len(self._configs)} agent configurations.")

    def get_config(self, agent_name: str) -> Optional[AgentConfig]:
        """
        Retrieves the configuration for a specific agent by name.
        """
        config = self._configs.get(agent_name)
        if not config:
            print(
                f"⚠️ Configuration for agent '{agent_name}' not found in registry.")
        return config


# Create a single, globally accessible instance of the registry.
AGENT_REGISTRY = AgentRegistry()

# --- Standalone Test Block ---
if __name__ == '__main__':
    print("\n--- Testing AgentRegistry ---")

    agent_names_to_test = [
        "MainRouter",
        "GoogleServicesAgent_Router",
        "GoogleCalendarTasksHandler",
        "GoogleMapsHandler",
        "GoogleMapsConclusionHandler",
        "WeatherAgent",
        "GeneralChatAgent",
        "NonExistentAgent"
    ]

    for name in agent_names_to_test:
        print(f"\n--- Retrieving config for: {name} ---")
        cfg = AGENT_REGISTRY.get_config(name)
        if cfg:
            print(f"  Name: {cfg.name}")
            print(f"  Model: {cfg.model}")
            print(f"  Prompt Name: {cfg.prompt_name}")
