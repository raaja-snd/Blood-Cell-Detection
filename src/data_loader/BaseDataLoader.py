
from abc import ABC, abstractmethod

class BaseDataLoader(ABC):

    @abstractmethod
    def load_dataset(self):
        """
        Load the dataset from the data source
        """
        pass