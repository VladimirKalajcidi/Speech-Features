import os
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
                          pipeline)


class PredictionPipeline:
    def __init__(self, audio):
        self.audio=audio
    
    def transcribe(self, model: str):
        device = "cpu"
        torch_dtype = torch.float32
        model_id = model
        #model_id = "artifacts/training/model"

        model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
        processor = WhisperProcessor.from_pretrained("openai/whisper-base", language="russian", task="transcribe")
        
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )

        text = pipe(self.audio)['text']
        
        return text
    

    def recognize_emotions(self):

        url = 'https://zenodo.org/record/6221127/files/w2v2-L-robust-12.6bc4a7fd-1.1.0.zip'
        cache_root = audeer.mkdir('cache')
        model_root = audeer.mkdir('model')

        archive_path = audeer.download_url(url, cache_root, verbose=True)
        audeer.extract_archive(archive_path, model_root)
        model = audonnx.load(model_root)

        sound = AudioSegment.from_mp3(self.audio)
        sound.export("audio.wav", format="wav")
        audio = "audio.wav"

        wav, fs = audiofile.read(audio)
        if fs != 16000:
            wav = audresample.resample(wav, fs, 16000)

        duration = librosa.get_duration(filename=audio)
        for i in range(wav.shape[0] // int(fs * duration)):
                pred = model(
                    wav[int((0 + i) * fs * duration):  int((i+1) * fs * duration)], fs)
                emotion = f"Arousal, dominance, valence #{i}: {pred['logits']}"
                return emotion
        if wav.shape[0] % int(fs * duration) != 0:
            pred = model(wav[-(wav.shape[0] % (fs*duration)):], fs)
            emotion = f"Arousal, dominance, valence #{i+1}: {pred['logits']}"
            return emotion
