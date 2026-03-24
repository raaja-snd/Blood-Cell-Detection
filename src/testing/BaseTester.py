from abc import ABC, abstractmethod

class BaseTester(ABC):

    def __init__(self):
        super().__init__()
    
    def setup(self):
        """
        Setup the test environment, model, logger and evaluators.
        """
        pass

    def run(self):
        """
        Run model on test data set and save metrices.
        """
        pass