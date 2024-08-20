import os
from speechRecognition.constants import Path, CONFIG_FILE_PATH, PARAMS_FILE_PATH
from speechRecognition.utils.common import read_yaml, create_directories
from speechRecognition.entity.config_entity import (DataIngestionConfig,
                                                    PrepareBaseModelConfig,
                                                    TrainingConfig)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_root=config.source_root,
            local_data_file=config.local_data_file, 
        )

        return data_ingestion_config
    

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            size=self.params.SIZE,
        )

        return prepare_base_model_config
    

    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        training_data = os.path.join(self.config.data_ingestion.root_dir, "data.hf")
        print(training_data)
        create_directories([
            Path(training.root_dir)
        ])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            size = params.SIZE,
            data_size=params.DATA_SIZE,
            max_steps=params.MAX_STEPS,
            language=params.LANGUAGE,
            task=params.TASK,
            metric=params.METRIC,
            learning_rate=params.LEARNING_RATE,
            generation_max_length=params.GENERATION_MAX_LENGTH,
            push_to_hub=params.PUSH_TO_HUB
        )

        return training_config