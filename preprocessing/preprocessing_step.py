from abc import ABC, abstractmethod


class IPreprocessingStep(ABC):

    @abstractmethod
    def preprocess_image(self, image):
        pass

    @abstractmethod
    def preprocess_mask(self, image, parameters):
        pass

    @abstractmethod
    def undo_preprocessing(self, image, parameters):
        pass