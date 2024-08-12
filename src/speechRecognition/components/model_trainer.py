from transformers import (WhisperTokenizer, 
                          WhisperFeatureExtractor, 
                          WhisperProcessor,
                          WhisperForConditionalGeneration,
                          Seq2SeqTrainingArguments,
                          Seq2SeqTrainer)

from speechRecognition.utils.common import DataCollatorSpeechSeq2SeqWithPadding
from speechRecognition.utils.common import compute_metrics
from speechRecognition.entity.config_entity import TrainingConfig
from dataclasses import dataclass
from datasets import load_dataset
from typing import Any, Dict, List, Union
from pathlib import Path

import torch
import evaluate



class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    
    def get_base_model(self):
        self.tokenizer = WhisperTokenizer.from_pretrained(self.config.size, 
                                                          language=self.config.language, 
                                                          task=self.config.task)
        
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(self.config.size)

        self.processor = WhisperProcessor.from_pretrained(self.config.size, 
                                                          language=self.config.language, 
                                                          task=self.config.task)
        
        self.model = WhisperForConditionalGeneration.from_pretrained(self.config.updated_base_model_path)


    def configure_trainig_arguments(self):

        self.model.generation_config.language = self.config.language
        self.model.generation_config.task = self.config.task
        self.model.generation_config.forced_decoder_ids = None
        
        self.data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=self.processor,
            decoder_start_token_id=self.model.config.decoder_start_token_id
        )
        
        self.metric = evaluate.load(self.config.metric)

        self.training_args = Seq2SeqTrainingArguments(
            output_dir=self.config.trained_model_path,
            learning_rate=1e-5,
            max_steps=self.config.max_steps,
            generation_max_length=self.config.generation_max_length,
            metric_for_best_model=self.config.metric,
            push_to_hub=self.config.push_to_hub
        )

        self.dataset = load_dataset(str(self.config.training_data))


    def configure_trainer(self):

        dataset = self.dataset
        data_size = self.config.data_size
        indices = [i for i in range(0, data_size)]

        dataset['train'] = dataset['train'].select(indices)
        dataset['test'] = dataset['train'].select(indices)

        self.trainer = Seq2SeqTrainer(
            args=self.training_args,
            model=self.model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=self.data_collator,
            compute_metrics=compute_metrics,
            tokenizer=self.processor.feature_extractor,
        )


    @staticmethod
    def save_model(path: Path, model: WhisperForConditionalGeneration):
        model.save_pretrained(path, from_pt=True)

    
    def train(self):
        
        self.trainer.train()

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )



