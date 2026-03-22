
from roboflow import Roboflow
from pathlib import Path
import os
import yaml
from dotenv import load_dotenv
from src.utils.logger import logger
from src.data_loader.BaseDataLoader import BaseDataLoader


class BcdDataIngestion(BaseDataLoader):
    """
    Data Ingestion class for BCD Dataset, which downloads the BCD dataset and saves them in the <project_root>/data/ path.
    """

    def __init__(self):
        super().__init__()
        self.parent_path = Path(__file__).resolve().parents[2]
        load_dotenv(str(self.parent_path/'env/.env'))
        self.project_config = self.__get_config_file()


    def __get_config_file(self):
        with open(self.parent_path/'src/config.yaml',encoding='utf-8') as config:
            project_config = yaml.safe_load(config)
        return project_config

    def load_dataset(self):
        """
        Load the BCD data set from the Roboflow workspace for the yolov8 model and saves them in <project_root>/data path.
        Requires the following evnironment variables:
        1. ROBOFLOW_API_KEY
        2. ROBOFLOW_WORKPLACE
        3. ROBOFLOW_PROJECT_ID

        """
        super().load_dataset()
        dataset_path = self.parent_path/'data'/f'{self.project_config['dataset_directory']}'
        if not dataset_path.exists():
            dataset_path.mkdir(parents=True, exist_ok=True)
        
        logger.debug('Downloading the BCCD dataset')
        rf = Roboflow(api_key=os.getenv('ROBOFLOW_API_KEY'))
        project = rf.workspace(os.getenv('ROBOFLOW_WORKSPACE'))\
                    .project(os.getenv('ROBOFLOW_PROJECT_ID'))
        version = project.version(1)
        version.download(self.project_config['model'],location=str(dataset_path),overwrite=True)
        logger.debug(f'Download Successful. Dataset saved in {str(dataset_path)}')

     