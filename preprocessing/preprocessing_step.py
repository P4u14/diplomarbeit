from abc import ABC, abstractmethod


class IPreprocessingStep(ABC):

    @abstractmethod
    def preprocess(self, image):
        pass

    @abstractmethod
    def undo_preprocessing(self, image):
        pass