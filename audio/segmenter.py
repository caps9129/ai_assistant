import json
from pathlib import Path
import torch
import queue
import sounddevice as sd
import threading
import numpy as np

# —————— VAD Configuration —————— #
# This block would be in your audio/config.json file
# {
#   "sample_rate": 16000,
#   "frame_duration_ms": 20,
#   "vad_threshold": 0.5,
#   "min_silence_duration_ms": 1000
# }
# ———————————————————————————————— #


class AudioSegmenter:
    def __init__(self):
        # Load configuration from JSON
        config_path = Path(__file__).parent / 'config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)

        self.sample_rate = config['sample_rate']
        self.frame_duration = config['frame_duration_ms']
        self.vad_threshold = config['vad_threshold']
        self.min_silence_duration_ms = config['min_silence_duration_ms']
        self.frame_bytes = self.sample_rate * self.frame_duration // 1000 * 2

        # Load the Silero VAD model
        try:
            self.model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                               model='silero_vad',
                                               force_reload=False)
            self.VADIterator = utils[3]
            print("✅ Silero VAD model loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading Silero VAD model: {e}")
            raise

        self.running = False
        self.frame_queue = queue.Queue()
        self.sentence_queue = queue.Queue()

    def start(self):
        """Starts the audio stream and the segmentation thread."""
        self.running = True

        def callback(indata, frames, time_info, status):
            if status:
                print("⚠️", status)
            pcm16 = (indata[:, 0] * 32767).astype(np.int16).tobytes()
            self.frame_queue.put(pcm16)

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1, dtype="float32",
            blocksize=self.frame_bytes // 2,
            callback=callback
        )
        self.stream.start()
        threading.Thread(target=self._segmenter_thread, daemon=True).start()
        print("✅ Audio segmenter started.")

    def _segmenter_thread(self):
        """
        Uses VADIterator for robust start/end detection and adds a custom
        silence timeout on top to handle long pauses correctly.
        """
        vad_iterator = self.VADIterator(
            self.model, threshold=self.vad_threshold, sampling_rate=self.sample_rate)

        audio_buffer = np.array([], dtype=np.float32)
        sentence_audio = np.array([], dtype=np.float32)

        in_speech = False
        is_potential_end = False
        silent_chunks_count = 0

        WINDOW_SIZE = 512
        SILENCE_CHUNKS_THRESHOLD = (
            self.min_silence_duration_ms / 1000) * (self.sample_rate / WINDOW_SIZE)

        while self.running:
            try:
                frame_bytes = self.frame_queue.get(timeout=0.1)
                frame_np = np.frombuffer(
                    frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                audio_buffer = np.concatenate([audio_buffer, frame_np])
            except queue.Empty:
                if is_potential_end:
                    self._finalize_sentence(sentence_audio)
                    in_speech, is_potential_end, sentence_audio = False, False, np.array(
                        [], dtype=np.float32)
                continue

            while len(audio_buffer) >= WINDOW_SIZE:
                processing_chunk = audio_buffer[:WINDOW_SIZE]
                audio_buffer = audio_buffer[WINDOW_SIZE:]

                speech_dict = vad_iterator(
                    processing_chunk, return_seconds=False)

                if not in_speech and speech_dict and 'start' in speech_dict:
                    # --- Speech Start ---
                    print("\n🎤 Speech detected...")
                    in_speech = True
                    sentence_audio = np.concatenate(
                        [sentence_audio, processing_chunk])

                elif in_speech:
                    # --- During Speech ---
                    sentence_audio = np.concatenate(
                        [sentence_audio, processing_chunk])

                    # --- MODIFIED LOGIC FOR ROBUST PAUSE HANDLING ---
                    if speech_dict and 'end' in speech_dict:
                        # VAD thinks it's an end; start the "potential end" countdown.
                        if not is_potential_end:
                            is_potential_end = True
                            silent_chunks_count = 0

                    if is_potential_end:
                        if speech_dict and 'start' in speech_dict:
                            # --- FIX: Speech has resumed ---
                            # VAD detected speech again, so cancel the "potential end".
                            print("... Resuming speech.")
                            is_potential_end = False
                            silent_chunks_count = 0
                        else:
                            # --- FIX: Only count silence when VAD is silent ---
                            # We are in a potential end, and the VAD is still silent.
                            silent_chunks_count += 1
                            if silent_chunks_count > SILENCE_CHUNKS_THRESHOLD:
                                # Silence timeout confirmed, finalize sentence
                                self._finalize_sentence(sentence_audio)
                                in_speech, is_potential_end, sentence_audio = False, False, np.array(
                                    [], dtype=np.float32)
                                vad_iterator.reset_states()  # Reset VAD for next utterance

    def _finalize_sentence(self, sentence_audio):
        """Helper to process a completed sentence."""
        if len(sentence_audio) > 0:
            sentence_bytes = (
                sentence_audio * 32767).astype(np.int16).tobytes()
            self.sentence_queue.put(sentence_bytes)
            duration = len(sentence_audio) / self.sample_rate
            print(
                f"✅ Sentence finalized after pause (duration: {duration:.2f}s).")

    def get_sentence(self):
        """Blocks until a complete sentence chunk is available."""
        return self.sentence_queue.get()

    def is_active(self):
        """Checks if the audio stream is currently active."""
        return self.running and hasattr(self, 'stream') and self.stream.active

    def stop(self):
        print("Stopping audio segmenter...")
        self.running = False
        if hasattr(self, 'stream') and self.stream.active:
            self.stream.stop()
            self.stream.close()
        print("Audio segmenter stopped.")
