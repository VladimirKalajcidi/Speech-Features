import os
from speechRecognition import logger
from datasets import load_dataset
from speechRecognition.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    def download_file(self)-> str:
        '''
        Fetch data from the url
        '''

        try: 
            dataset_root = self.config.source_root
            download_dir = self.config.local_data_file
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logger.info(f"Downloading data from {dataset_root} into file {download_dir}")

            dataset = load_dataset("vladimir7542/for_whisper_ft1_prepared")
            dataset.save_to_disk(download_dir)

            logger.info(f"Downloaded data from {dataset_root} into file {download_dir}")

        except Exception as e:
            raise e
