import numpy as np

from segmenter.atlas.voter.segmentation_voter import ISegmentationVoter


class WeightedMajorityVoter(ISegmentationVoter):

    def __init__(self, scheme: str = "normalize", temperature: float = 0.02, threshold: float = 0.5):
        """
        Initializes the WeightedMajorityVoter with a specific voting scheme, temperature, and threshold.

        :param scheme: The voting scheme to use. Options are "normalize" or "softmax".
        :param temperature: Temperature parameter for softmax scaling.
        :param threshold: Threshold for binarizing the final vote.
        """
        self.scheme = scheme
        self.temperature = temperature
        self.threshold = threshold

    @staticmethod
    def normalize_weights(scored_atlases):
        """
        Normalize the scores of the scored classes to sum to 1.

        :param scored_atlases: List of scored classes with their scores.
        :return: Normalized weights.
        """
        weights = np.array([scored_atlas.score for scored_atlas in scored_atlases])
        return weights / np.sum(weights) if np.sum(weights) > 0 else weights

    def softmax_weights(self, scored_atlases):
        """
        Compute softmax-normalized weights from the scores of the scored atlases.
        :param scored_atlases: List of scored classes with their scores.
        :return: Softmax-normalized weights.
        Higher scores yield exponentially higher weights. The temperature parameter controls the sharpness of the distribution.
        Lower temperature (e.g., <1) makes the weighting more selective (sharper focus on the highest values).
        Higher temperature (e.g., >1) makes the weighting more uniform (weights are distributed more equally).
        """
        weights = np.array([scored_atlas.score for scored_atlas in scored_atlases])
        e_x = np.exp((weights - np.max(weights)) / self.temperature)
        return e_x / (e_x.sum() + 1e-8)

    def compute_weights(self, scored_atlases):
        """
        Compute the weights for the scored atlases based on the specified voting scheme.
        :param scored_atlases: List of scored classes with their scores.
        :return: Weights for each atlas based on the voting scheme.
        Raises ValueError if an unknown voting scheme is specified.
        """
        if self.scheme == "normalize":
            return self.normalize_weights(scored_atlases)
        elif self.scheme == "softmax":
            return self.softmax_weights(scored_atlases)
        else:
            raise ValueError(f"Unknown voting scheme: {self.scheme}. Use 'normalize' or 'softmax'.")

    def vote(self, scored_atlases):
        """
        Perform weighted majority voting on the scored atlases.
        :param scored_atlases: List of scored classes with their scores.
        :return: A binary mask where each pixel is set if the weighted sum exceeds the threshold.
        The voting is done by computing the weighted sum of the masks from the scored atlases.
        Each mask is weighted according to the computed weights, and the final mask is binarized based on the threshold.
        The result is a binary mask where each pixel is set if the weighted sum exceeds the threshold (default 0.5).
        """
        weights = self.compute_weights(scored_atlases)
        masks = np.stack([scored_atlas.atlas.preprocessed_mask for scored_atlas in scored_atlases], axis=0) / 255
        votes = np.tensordot(weights, masks, axes=([0], [0]))
        return (votes > self.threshold).astype(np.uint8)