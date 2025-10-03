import numpy as np

from segmenter.atlas.voter.segmentation_voter import ISegmentationVoter


class MajorityVoter(ISegmentationVoter):
    """
    Segmentation voter that applies majority voting to a set of atlas masks.
    For each pixel, the class with the majority of votes across all atlases is selected as the final segmentation.
    """

    def vote(self, scored_atlases):
        """
        Perform majority voting on the masks of the provided scored atlases.
        Args:
            scored_atlases (list): List of scored atlas objects, each with an atlas containing a preprocessed_mask.
        Returns:
            np.ndarray: Binary mask resulting from majority voting (same shape as input masks).
        """
        masks = np.stack([scored_atlas.atlas.preprocessed_mask for scored_atlas in scored_atlases], axis=0) / 255
        votes = np.sum(masks > 0.5, axis=0)
        return (votes > (len(masks) // 2)).astype(np.uint8)