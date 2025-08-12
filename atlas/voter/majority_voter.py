import numpy as np

from atlas.voter.segmentation_voter import ISegmentationVoter


class MajorityVoter(ISegmentationVoter):

    def vote(self, scored_atlases):
        masks = np.stack([scored_atlas.atlas.preprocessed_mask for scored_atlas in scored_atlases], axis=0)
        votes = np.sum(masks > 0.5, axis=0)
        return (votes > (len(masks) // 2)).astype(np.uint8)