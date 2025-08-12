from abc import ABC, abstractmethod

from atlas.atlas import Atlas


class ISegmentationVoter(ABC):

    @abstractmethod
    def vote(self, scored_atlases):
        pass