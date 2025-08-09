from abc import ABC, abstractmethod
import threading


class TTSEngineBase(ABC):
    """
    An abstract base class that defines the interface for all TTS engines.
    """

    def __init__(self, voice: str, stop_event: threading.Event, **kwargs):
        """
        Initializes the engine.

        Args:
            voice (str): The voice model to use.
            stop_event (threading.Event): An event to signal interruption.
        """
        self.voice = voice
        self.stop_event = stop_event
        print(f"[{self.__class__.__name__}] Initialized for voice '{self.voice}'.")

    @abstractmethod
    def say(self, text: str):
        """
        Synthesizes and plays the given text. This is a blocking call that
        should only return after playback is finished or interrupted.
        """
        pass

    @abstractmethod
    def stop(self):
        """
        Forcibly stops any ongoing playback.
        """
        pass
