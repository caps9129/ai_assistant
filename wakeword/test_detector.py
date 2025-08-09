import unittest
from unittest.mock import patch, MagicMock

from wakeword.detector import WakeWordDetector

class TestWakeWordDetector(unittest.TestCase):
    @patch('wakeword.detector.sd.Stream')
    @patch('wakeword.detector.Model')
    def test_start_and_stop(self, MockModel, MockStream):
        # Initialize detector with dummy model paths
        wake_models = ['dummy.onnx']
        melspec = 'melspec.onnx'
        embed = 'embed.onnx'
        detector = WakeWordDetector(
            wakeword_models=wake_models,
            threshold=0.7,
            melspec_model_path=melspec,
            embedding_model_path=embed
        )
        # Model should be initialized with given paths
        MockModel.assert_called_once_with(
            wakeword_models=wake_models,
            inference_framework='onnx',
            melspec_model_path=melspec,
            embedding_model_path=embed
        )
        # Before start, stream should be None
        self.assertIsNone(detector.stream)

        # Start should create and start the stream
        detector.start(device=1)
        MockStream.assert_called_once_with(
            device=1,
            samplerate=detector.rate,
            channels=(detector.channels, detector.channels),
            dtype='int16',
            blocksize=detector.blocksize,
            callback=detector._duplex_callback
        )
        detector.stream.start.assert_called_once()
        self.assertIsNotNone(detector.stream)

        # Stop should stop and close the stream on the original instance
        stream_instance = detector.stream
        detector.stop()
        stream_instance.stop.assert_called_once()
        stream_instance.close.assert_called_once()
        self.assertIsNone(detector.stream)

    @patch('wakeword.detector.sd.Stream')
    @patch('wakeword.detector.Model')
    def test_double_start(self, MockModel, MockStream):
        # Provide required model paths
        detector = WakeWordDetector(
            wakeword_models=['dummy.onnx'],
            melspec_model_path='melspec.onnx',
            embedding_model_path='embed.onnx'
        )
        # Simulate already running
        detector.stream = MagicMock()
        with patch('builtins.print') as mock_print:
            detector.start()
            mock_print.assert_called_with("Stream is already running.")

    @patch('wakeword.detector.sd.Stream')
    @patch('wakeword.detector.Model')
    def test_stop_not_running(self, MockModel, MockStream):
        # Provide required model paths
        detector = WakeWordDetector(
            wakeword_models=['dummy.onnx'],
            melspec_model_path='melspec.onnx',
            embedding_model_path='embed.onnx'
        )
        detector.stream = None
        with patch('builtins.print') as mock_print:
            detector.stop()
            mock_print.assert_called_with("Stream is not running.")

if __name__ == '__main__':
    unittest.main()
