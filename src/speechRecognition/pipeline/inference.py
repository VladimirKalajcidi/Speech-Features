'''import os
import torch
import audeer
import audonnx
import numpy as np
import audiofile
import librosa
import audresample
from pydub import AudioSegment
from transformers import (WhisperProcessor,
                          WhisperForConditionalGeneration,
                          pipeline)'''
import whisperx


class PredictionPipeline:
    def __init__(self, audio):
        self.audio=audio
    
    def transcribe(self, model: str, token: str):
        device = "cpu" 
        audio_file = self.audio
        batch_size = 16
        compute_type = "int8"

        model = whisperx.load_model(model, device, compute_type=compute_type)

        audio = whisperx.load_audio(audio_file)
        result = model.transcribe(audio, batch_size=batch_size)

        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)


        diarize_model = whisperx.DiarizationPipeline(use_auth_token=token, device=device)

        diarize_segments = diarize_model(audio, min_speakers=2, max_speakers=2)

        result = whisperx.assign_word_speakers(diarize_segments, result)
        #print(diarize_segments)
        
        return result["segments"] # segments are now assigned speaker IDs
