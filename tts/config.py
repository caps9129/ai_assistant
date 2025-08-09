from dataclasses import dataclass
from typing import Dict


@dataclass
class TTSConfig:
    """A structured class to hold the configuration for a single TTS engine."""
    provider: str
    voice: str
    model: str = "tts-1"  # Default model
    sample_rate: int = 24000

# --- Main TTS Configuration ---
# This is the central dictionary that the TTS router will import and use.


TTS_CONFIG: Dict[str, any] = {
    # [KEY SETTING] Change this value to switch between TTS providers
    # Valid options: "edge", "openai"

    # --- Provider-Specific Settings ---
    "providers": {
        "edge": {
            "zh": TTSConfig(
                provider="edge",
                voice="zh-TW-HsiaoChenNeural",
                sample_rate=24000
            ),
            "en": TTSConfig(
                provider="edge",
                voice="en-US-AriaNeural",
                sample_rate=24000
            )
        },
        "openai": {
            # For OpenAI, Chinese and English can use the same voice model
            "zh": TTSConfig(
                provider="openai",
                voice="nova",  # Voices: alloy, echo, fable, onyx, nova, shimmer
                model="tts-1",  # Can be "tts-1" or "tts-1-hd" for higher quality
                sample_rate=24000
            ),
            "en": TTSConfig(
                provider="openai",
                voice="nova",
                model="tts-1",
                sample_rate=24000
            )
        },
        "openai-sys": {
            # For OpenAI, Chinese and English can use the same voice model
            "zh": TTSConfig(
                provider="openai",
                voice="nova",  # Voices: alloy, echo, fable, onyx, nova, shimmer
                model="tts-1",  # Can be "tts-1" or "tts-1-hd" for higher quality
                sample_rate=24000
            ),
            "en": TTSConfig(
                provider="openai",
                voice="nova",
                model="tts-1",
                sample_rate=24000
            )
        }
    }
}
