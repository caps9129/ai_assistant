# transcribers/router.py

# Assuming your other classes are in sibling files or configured in __init__.py
from .engine import Transcriber, FasterWhisperTranscriber, GoogleAPITranscriber, OpenAITranscriber
# Assuming your Language Enum is in a root-level config.py
from config import Language
import numpy as np
from dotenv import load_dotenv
load_dotenv()


class AsrRouter(Transcriber):
    """
    An intelligent router that directs audio to the best transcription engine.
    - Uses Whisper for robust, offline language detection.
    - Routes to language-specific Google ASR for speed.
    - Falls back to Whisper for transcription of other languages.
    """

    def __init__(self, whisper_model_size="tiny", enable_openai: bool = False):
        print("Initializing ASR Router...")
        # 1. A lightweight Whisper model dedicated to language detection and as a fallback
        self.whisper_engine = FasterWhisperTranscriber(
            whisper_model_size,
            device="cpu",
            compute_type="int8"
        )

        # 2. Specialized online transcribers for common languages
        self.en_transcriber = GoogleAPITranscriber(language=Language.ENGLISH)
        self.zh_transcriber = GoogleAPITranscriber(language=Language.CHINESE)
        self.openai_transcriber = OpenAITranscriber(
            model="gpt-4o-mini-transcribe")
        self.enable_openai = enable_openai
        print("✅ ASR Router is ready.")

    def transcribe(self, audio_bytes: bytes) -> dict:
        """
        Detects the language from audio and routes it to the appropriate transcriber.

        Args:
            audio_bytes: Raw PCM audio data.
            enable_openai: using openai transcriber

        Returns:
            A dictionary containing the transcription result.
        """

        if self.enable_openai:
            return self.openai_transcriber.transcribe(audio_bytes)

        # Step 1: Detect the language using Whisper's robust engine
        lang_result = self.whisper_engine.detect_language(audio_bytes)
        if lang_result.get('error'):
            print(f"⚠️  Could not detect language: {lang_result['error']}")
            # Fallback to Whisper for transcription if detection fails
            return self.whisper_engine.transcribe(audio_bytes)

        lang_code = lang_result.get('language')

        # Step 2: Route to the appropriate transcriber based on the language
        if lang_code == 'zh':
            print("--> Routing to Chinese (Google) transcriber...")
            return self.zh_transcriber.transcribe(audio_bytes)
        else:
            print("--> Routing to English (Google) transcriber...")
            return self.en_transcriber.transcribe(audio_bytes)
        # else:
        #     print(f"--> Language '{lang_code}' not specialized, falling back to Whisper...")
        #     # For any other language, use Whisper's multi-language capabilities
        #     return self.whisper_engine.transcribe(audio_bytes, language_code=lang_code)


if __name__ == '__main__':

    # This block allows you to test this file directly.
    # To run, execute from your project root: python -m transcribers.router
    print("--- Running AsrRouter standalone test ---")

    # 1. Initialize the router
    # Using 'tiny' model for a quick test
    asr_router = AsrRouter(whisper_model_size="tiny")

    # 2. Create a dummy audio chunk (e.g., 3 seconds of silence)
    sample_rate = 16000
    silent_audio = np.zeros(sample_rate * 3, dtype=np.int16).tobytes()

    # 3. Test the transcription process
    # The language detector will likely default to English ('en') on silent audio
    print("\nTesting with silent audio...")
    result = asr_router.transcribe(silent_audio)

    print("\n--- Test Result ---")
    print(result)
    print("-------------------")
