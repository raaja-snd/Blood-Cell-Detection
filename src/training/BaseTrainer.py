from abc import ABC, abstractmethod

class BaseTrainer(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def setup(self):
        """
        Setup the training arguments and the model trainer. This is called before calling the run on the trainer.
        """
        pass

    @abstractmethod
    def run(self, experiment_name = ''):
        """
        Run the trainer and track metrices.
        """
        self.experiment_name = experiment_name
    