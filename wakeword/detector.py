import sounddevice as sd
from openwakeword.model import Model
import threading
from pathlib import Path
import json

class WakeWordDetector:
    """
    A robust, full-duplex wake word detector with callback support.
    """
    def __init__(
        self,
        threshold=0.5,
        samplerate=16000,
        channels=1,
        blocksize=1280,
        inference_framework='onnx',
        on_detection=None  ## NEW: Callback function to be executed on detection
    ):
        
        config_path = Path(__file__).parent / 'config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)

        self.rate = samplerate
        self.channels = channels
        self.blocksize = blocksize
        self.threshold = threshold
        self.on_detection = on_detection  ## NEW: Store the callback
        self.stream = None
        # print(31, config['model'])
        self.model = Model(
            wakeword_models=[config['model']],
            inference_framework=inference_framework,
            melspec_model_path=config['melspec_model'],
            embedding_model_path=config['embedding_model'],
        )

    def _duplex_callback(self, indata, outdata, frames, time, status):
        if status.input_overflow:
            return
        outdata.fill(0)
        
        audio = indata[:, 0]
        self.model.predict(audio)

        ## MODIFIED: Call the callback instead of printing directly
        for mdl in self.model.prediction_buffer:
            score = self.model.prediction_buffer[mdl][-1]
            if score > self.threshold:
                print(f"✅ Detected '{mdl}' with score {score:.3f}")
                if self.on_detection:
                    self.on_detection() # Execute the provided callback function
                break # Stop checking other models once one is detected

    def is_active(self):
        ## NEW: Helper method to check stream status
        return self.stream is not None and self.stream.active

    def wait_for_wakeword(self, device=None):
        ## NEW: A simple, blocking method for easy use
        """
        Starts the detector and blocks until the wakeword is detected.
        Automatically handles starting and stopping the stream.
        """
        detection_event = threading.Event()
        
        # Temporarily set the callback to our event
        original_callback = self.on_detection
        self.on_detection = detection_event.set
        
        print("🔊 Waiting for wakeword...")
        self.start(device=device)
        
        # Wait here until the callback sets the event
        detection_event.wait()
        
        self.stop()
        print("Wakeword detected, detector stopped.")
        
        # Restore original callback
        self.on_detection = original_callback


    def start(self, device=None):
        if self.is_active():
            print("Stream is already running.")
            return
        
        # Reset previous predictions before starting
        self.model.reset()

        self.stream = sd.Stream(
            device=device,
            samplerate=self.rate,
            channels=(self.channels, self.channels),
            dtype='int16',
            blocksize=self.blocksize,
            callback=self._duplex_callback,
        )
        self.stream.start()
        print("🔊 WakeWordDetector started.")
    
    def stop(self):
        if not self.is_active():
            # print("Stream is not running.") # Usually not needed to print this
            return
        
        self.stream.stop()
        self.stream.close()
        self.stream = None
        print("🛑 WakeWordDetector stopped.")


if __name__ == "__main__":
    import argparse

    # --- Example of using the new callback system ---
    def handle_detection():
        print("--> Main program was notified of the detection!")
        # In a real app, you might set a global event here

    parser = argparse.ArgumentParser(description="Full-duplex Wake Word Detector")
    # ... (your argparse code remains the same)
    parser.add_argument('--model-path', type=str, required=True, help="Path to your wake word ONNX model")
    parser.add_argument('--melspec-model', type=str, required=True, help="Path to the melspectrogram ONNX model")
    parser.add_argument('--embedding-model', type=str, required=True, help="Path to the embedding ONNX model")
    parser.add_argument('--threshold', type=float, default=0.5, help="Detection threshold (0.0-1.0)")
    parser.add_argument('--device', type=str, default=None, help="Sounddevice device specifier (None for default)")
    args = parser.parse_args()

    detector = WakeWordDetector(
        wakeword_models=[args.model_path],
        threshold=args.threshold,
        melspec_model_path=args.melspec_model,
        embedding_model_path=args.embedding_model,
        on_detection=handle_detection # Pass our handler function
    )

    try:
        # Example 1: Non-blocking detection with callback
        print("\n--- Running non-blocking detection for 10 seconds ---")
        detector.start(device=args.device)
        import time
        time.sleep(10)
        detector.stop()

        # Example 2: Blocking detection
        print("\n--- Running blocking detection (will stop after first detection) ---")
        detector.wait_for_wakeword(device=args.device)
        print("Blocking call finished.")

    except KeyboardInterrupt:
        pass
    finally:
        if detector.is_active():
            detector.stop()