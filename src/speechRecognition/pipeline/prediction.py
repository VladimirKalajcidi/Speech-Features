import os
import torch
from transformers import (WhisperProcessor,
                          WhisperForConditionalGeneration,
                          pipeline)



class PredictionPipeline:
    def __init__(self, audio):
        self.audio=audio
    
    def predict(self):
        device = "cpu"
        torch_dtype = torch.float32
        model_id = "artifacts/training/model"

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

        result = pipe(self.audio)
        
        return result
    