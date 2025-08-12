from abc import ABC, abstractmethod

from atlas.atlas import Atlas


class IAtlasSelector(ABC):
    @abstractmethod
    def select_atlases(self, atlases: [Atlas], target_image, num_atlases_to_select) -> [Atlas]:
        pass