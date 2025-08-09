from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from string import Template
from utils.location import LocationManager


class PromptManager:
    _instance = None
    _templates = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PromptManager, cls).__new__(cls)
            cls._instance._load_all_templates()
            cls._instance.location_manager = LocationManager()
        return cls._instance

    def _load_template(self, file_path: Path) -> str:
        """Loads a single prompt file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            print(f"⚠️ Prompt file not found: {file_path}")
            return ""

    def _load_all_templates(self):
        """Loads all .txt prompt files from the prompts directory and subdirectories."""
        prompts_dir = Path(__file__).parent
        for prompt_file in prompts_dir.rglob("*.txt"):
            key = str(prompt_file.relative_to(prompts_dir)).replace(
                ".txt", "").replace("\\", "/")
            self._templates[key] = self._load_template(prompt_file)
        print(f"✅ PromptManager loaded {len(self._templates)} templates.")

    def get(self, template_name: str) -> str:
        """
        Gets a prompt and formats it with any required dynamic data using string.Template.
        """
        template_str = self._templates.get(template_name)
        if not template_str:
            raise ValueError(f"Prompt template '{template_name}' not found.")

        prompt_template = Template(template_str)

        # This dictionary will hold all the dynamic data to be injected.
        data_to_inject = {}

        # --- Check for and add current_date ---
        if "${current_date}" in template_str:
            timezone_name = self.location_manager.get_current_timezone_name()
            tz = ZoneInfo(timezone_name) if timezone_name else None
            current_time = datetime.now(tz)
            data_to_inject['current_date'] = current_time.strftime(
                '%A, %B %d, %Y, %I:%M %p %Z')

        # --- NEW: Check for and add current_location ---
        if "${current_location}" in template_str:
            location_info = self.location_manager.get_location_info()
            if location_info:
                # Create a clean, human-readable location string
                city = location_info.get('city', '')
                region = location_info.get('region', '')
                country = location_info.get('country', '')
                # Filter out empty parts and join
                location_parts = [part for part in [
                    city, region, country] if part]
                data_to_inject['current_location'] = ", ".join(location_parts)
            else:
                # Fallback
                data_to_inject['current_location'] = "an unknown location"

        # .substitute() will replace all placeholders for which it finds a key in the dictionary.
        return prompt_template.substitute(data_to_inject)


# Create a single, globally accessible instance
PROMPT_MANAGER = PromptManager()
