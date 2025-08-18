from abc import ABC, abstractmethod


class IPreprocessingStep(ABC):

    @abstractmethod
    def preprocess(self, image):
        pass

    @abstractmethod
    def preprocess_with_parameters(self, image, parameters):
        pass

    @abstractmethod
    def undo_preprocessing(self, image, params):
        pass