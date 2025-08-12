from abc import ABC, abstractmethod


class ISegmentationVoter(ABC):

    @abstractmethod
    def vote(self, scored_atlases):
        pass