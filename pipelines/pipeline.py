#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pickle
from configparser import ConfigParser

from pipelines.model import AVSR
from pipelines.data.data_module import AVSRDataLoader
import pyttsx3


class InferencePipeline(torch.nn.Module):
    def __init__(self, config_filename, detector="retinaface", face_track=False, device="cuda:0"):
        super(InferencePipeline, self).__init__()
        assert os.path.isfile(config_filename), f"config_filename: {config_filename} does not exist."

        config = ConfigParser()
        config.read(config_filename)

        # modality configuration
        modality = config.get("input", "modality")

        self.modality = modality
        # data configuration
        input_v_fps = config.getfloat("input", "v_fps")
        model_v_fps = config.getfloat("model", "v_fps")

        # model configuration
        model_path = config.get("model","model_path")
        model_conf = config.get("model","model_conf")

        # language model configuration
        rnnlm = config.get("model", "rnnlm")
        rnnlm_conf = config.get("model", "rnnlm_conf")
        penalty = config.getfloat("decode", "penalty")
        ctc_weight = config.getfloat("decode", "ctc_weight")
        lm_weight = config.getfloat("decode", "lm_weight")
        beam_size = config.getint("decode", "beam_size")

        self.dataloader = AVSRDataLoader(modality, speed_rate=input_v_fps/model_v_fps, detector=detector)
        self.model = AVSR(modality, model_path, model_conf, rnnlm, rnnlm_conf, penalty, ctc_weight, lm_weight, beam_size, device)
        if face_track and self.modality in ["video", "audiovisual"]:
            if detector == "mediapipe":
                from pipelines.detectors.mediapipe.detector import LandmarksDetector
                self.landmarks_detector = LandmarksDetector()
            if detector == "retinaface":
                from pipelines.detectors.retinaface.detector import LandmarksDetector
                self.landmarks_detector = LandmarksDetector(device="cuda:0")
        else:
            self.landmarks_detector = None


    def process_landmarks(self, data_filename, landmarks_filename):
        if self.modality == "audio":
            return None
        if self.modality in ["video", "audiovisual"]:
            if isinstance(landmarks_filename, str):
                landmarks = pickle.load(open(landmarks_filename, "rb"))
            else:
                landmarks = self.landmarks_detector(data_filename)
            return landmarks


    def forward(self, data_filename, landmarks_filename=None):
        assert os.path.isfile(data_filename), f"data_filename: {data_filename} does not exist."
        landmarks = self.process_landmarks(data_filename, landmarks_filename)
        data = self.dataloader.load_data(data_filename, landmarks)
        transcript = self.model.infer(data)
        return transcript

    @staticmethod
    def text_to_speech(corrected_text, output_audio_path):
        """
        Converts corrected text into a .wav audio file using Microsoft SpeechT5.

        Args:
            corrected_text (str): The corrected text to convert to speech.
            output_audio_path (str): Path to save the generated .wav file.
        """
        try:
            from transformers import pipeline
            from datasets import load_dataset
            import soundfile as sf
            import torch
            import os

            print("Initializing SpeechT5 TTS...")
            
            # Determine device
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
                
            # Initialize the pipeline
            synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts", device=device)
            
            # Load xvector containing speaker's voice characteristics from a dataset
            embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
            speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
            
            # Generate speech
            speech = synthesiser(corrected_text, forward_params={"speaker_embeddings": speaker_embedding})
            
            # Save to file
            sf.write(output_audio_path, speech["audio"], samplerate=speech["sampling_rate"])
            
            print(f"Audio file saved at: {output_audio_path}")
            return output_audio_path
            
        except Exception as e:
            print(f"Error using SpeechT5 TTS: {e}")
            print("Falling back to pyttsx3...")
            
            # Fallback to pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 1.0)
            engine.save_to_file(corrected_text, output_audio_path)
            engine.runAndWait()
            
            print(f"Audio file saved at: {output_audio_path} (via fallback)")
            return output_audio_path


if __name__ == "__main__":
    # Test the text_to_speech function
    corrected_text = "Hello, this is a test of the text-to-speech functionality."
    output_audio_path = "test_audio.wav"

    # Call the function to generate the audio file
    InferencePipeline.text_to_speech(corrected_text, output_audio_path)
