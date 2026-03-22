from pathlib import Path
import mlflow
from pathlib import Path
import yaml
from ultralytics import YOLO
from src.training.BaseTrainer import BaseTrainer
from src.utils.logger import logger
from src.utils.device import get_device

class BcdYoloTrainer(BaseTrainer):

    def __init__(self, name = 'YoloTrainer'):
        super().__init__()
        self.parent_path = Path(__file__).resolve().parents[2]
        with open(self.parent_path/'src/config.yaml',encoding='utf-8') as c:
            config = yaml.safe_load(c)
        self.training_config = config['training']
        self.model_config = config['model']
        self.mlflow_config = config['mlflow']
        self.name = name
        logger.info(f'Initialized {name}')
    
    def setup(self,config_overrides = {}):
        super().setup()
        self.training_config = {**self.training_config,**config_overrides}
        
        if get_device() == 'cpu':
            logger.info('No cuda device found. rolling back to cpu')
        self.training_config['device'] = get_device()

        #Construct dataset path relative to project root directory
        self.training_config['data'] = str(self.parent_path/f'{self.training_config['data']}')

        self.model = YOLO(model=self.model_config['kind'],task='detect')

        self.tracking_uri = self.parent_path/f'{self.mlflow_config['yolo']['tracking_uri']}'
        if not self.tracking_uri.exists():
            self.tracking_uri.mkdir(parents=True,exist_ok=True)

        logger.info(f'{self.name} setup complete')
    
    def run(self):
        super().run()
        logger.info('Starting trainer and setting up experimental tracking')
        mlflow.set_tracking_uri(self.tracking_uri.as_uri())
        mlflow.set_experiment(self.mlflow_config['yolo']['experiment_name'])
        mlflow.set_tags({
            "project":"blood-cell-detection",
            "model":"yolov8"
        })

        logger.info('Training started')

        self._track_params()

        result = self.model.train(**self.training_config)

        self._track_metrics(result.results_dict)
        self._save_model(result)
        logger.info('Training complete!')
    
    def _track_params(self):
        mlflow.log_params({
            'lr':self.training_config['lr0'],
            'epochs': self.training_config['epochs'],
            'batch':self.training_config['batch'],
            'weight_decay': self.training_config['weight_decay'],
            'cls':self.training_config['cls']
        })
    
    def _track_metrics(self, metrics):
        mlflow.log_metric("mAP50",     metrics.get("metrics/mAP50(B)", 0))
        mlflow.log_metric("mAP50-95",  metrics.get("metrics/mAP50-95(B)", 0))
        mlflow.log_metric("precision", metrics.get("metrics/precision(B)", 0))
        mlflow.log_metric("recall",    metrics.get("metrics/recall(B)", 0))

    def _save_model(self,result):
        best_model_weights = Path(result.save_dir)/'weights'/'best.pt'

        if not best_model_weights:
            logger.warning('Best model weights not found')

        model_path = self.parent_path/f'{self.model_config['model_directory']}/bcd_yolo.pt'
        if not model_path.parent.exists():
            model_path.parent.mkdir(parents=True,exist_ok=True)
        
        import shutil
        shutil.copy2(str(best_model_weights), str(model_path))

        logger.info(f'Best model saved to {model_path}')


