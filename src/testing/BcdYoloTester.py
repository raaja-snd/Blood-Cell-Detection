from pathlib import Path
import yaml
from src.utils.logger import logger
from src.utils.device import get_device
from ultralytics import YOLO
from src.testing.BaseTester import BaseTester
import mlflow

class BcdYoloTester(BaseTester):

    def __init__(self):
        super().__init__()
        self.parent_path = Path(__file__).resolve().parents[2]
        with open(self.parent_path/'src/config.yaml',encoding='utf-8') as c:
            config = yaml.safe_load(c)
        
        self.test_config = config['test']
        self.model_config = config['model']
        self.mlflow_config = config['mlflow']['yolo-test']
        self.dataset_directory = self.parent_path/f'{config['dataset_directory']}'
        self.output_directory = self.parent_path/f'{self.test_config['parent']}'
    
    def setup(self, config_overrides = {}):
        """
        Setup the Yolo test model. The model is tested using predifined test configuration. 
        The configuration is provided in the config.yaml file in the src/ directory. To 
        override some of the configuation, use the config_overrides argument.

        Args:
            config_overrides(dict): The parameters to override for testing.
        """
        super().setup()

        self.test_config = {**self.test_config['parameters'],**config_overrides}

        if not self.output_directory.exists():
            self.output_directory.mkdir(parents=True,exist_ok=True)

        self.test_config['project'] = str(self.output_directory.parent)
        self.test_config['name'] = str(self.output_directory.name)
        

        self.model_path = self.parent_path/f'{self.model_config['model_directory']}'/'bcd_yolo.pt'
        self.model = YOLO(self.model_path,task='detect')

        self.mlflow_tracking_uri = self.parent_path/f'{self.mlflow_config['tracking_uri']}'
        if not self.mlflow_tracking_uri.exists():
            self.mlflow_tracking_uri.mkdir(parents=True, exist_ok=True)

        logger.info("BCD Yolo Test seup complete")
    
    def run(self):
        """
        Run the YOLO tester. Each run tracks the metrices usin the MlFLow. Test results
        are saved in src/testing/results directory.

        """
        super().run()
        logger.info('Starting yolo tester and setting up experimental tracking')

        mlflow.set_tracking_uri(self.mlflow_tracking_uri.as_uri())
        mlflow.set_experiment(self.mlflow_config['experiment_name'])
        mlflow.set_tags({
            "project":"blood-cell-detection",
            "model":"yolov8"
        })

        logger.info('Yolo BCD Test Started')
        result = self.model.val(split='test',**self.test_config)

        logger.info('Testing complete')
        self.log_metrics(result.results_dict)

    
    def log_metrics(self, metrics):
        mlflow.log_metric("mAP50",     metrics.get("metrics/mAP50(B)", 0))
        mlflow.log_metric("mAP50-95",  metrics.get("metrics/mAP50-95(B)", 0))
        mlflow.log_metric("precision", metrics.get("metrics/precision(B)", 0))
        mlflow.log_metric("recall",    metrics.get("metrics/recall(B)", 0))
