import os
import urllib.request as request
from zipfile import ZipFile
import whisper
import torch
from transformers import WhisperForConditionalGeneration
from pathlib import Path
from speechRecognition.entity.config_entity import (PrepareBaseModelConfig)



class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = WhisperForConditionalGeneration.from_pretrained(self.config.size)

        self.save_model(path=self.config.base_model_path, model=self.model)

        self.save_model(path=self.config.updated_base_model_path, model=self.model)
    

    @staticmethod
    def save_model(path: Path, model):
        model.save_pretrained(path, from_pt=True)