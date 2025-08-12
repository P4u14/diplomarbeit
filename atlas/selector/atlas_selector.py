from abc import ABC, abstractmethod


class IAtlasSelector(ABC):
    @abstractmethod
    def select_atlases(self, atlases, target_image, n):
        pass