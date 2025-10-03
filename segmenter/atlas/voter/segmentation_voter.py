from abc import ABC, abstractmethod


class ISegmentationVoter(ABC):
    """
    Interface for segmentation voting strategies in atlas-based segmentation workflows.
    All segmentation voter classes should inherit from this interface and implement the vote method.
    """

    @abstractmethod
    def vote(self, scored_atlases):
        """
        Perform voting on the masks of the provided scored atlases to produce a final segmentation mask.

        Args:
            scored_atlases (list): List of scored atlas objects, each with an atlas containing a preprocessed_mask.

        Returns:
            np.ndarray: Final segmentation mask after voting.
        """
        pass