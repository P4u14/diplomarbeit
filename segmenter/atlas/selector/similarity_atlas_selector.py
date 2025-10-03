from skimage.metrics import structural_similarity
from typing_extensions import override

from segmenter.atlas.atlas_score import AtlasScore
from segmenter.atlas.selector.base_atlas_selector import BaseAtlasSelector


class SimilarityAtlasSelector(BaseAtlasSelector):
    """
    Atlas selector that selects atlases based on similarity to the target image.
    Atlases are ranked by a similarity score, and the top N are selected.
    """

    @override
    def select_atlases(self, atlases, target_image, n):
        """
        Select the top N atlases most similar to the target image.

        Args:
            atlases (list): List of available Atlas objects.
            target_image: The target image for which atlases are to be selected.
            n (int): Number of atlases to select.

        Returns:
            list: List of selected Atlas objects.
        """
        scored_atlases = self.score_atlases(atlases, target_image)
        return scored_atlases[:n]

    def score_atlases(self, atlases, target_image):
        """
        Compute the similarity scores of atlases compared to the target image.

        Args:
            atlases (list): List of available Atlas objects.
            target_image: The target image for which atlases are to be scored.

        Returns:
            list: List of AtlasScore objects containing atlases and their corresponding scores.
        """
        target_image_gray = self.to_gray(target_image.preprocessed_image)
        results = []
        for atlas in atlases:
            atlas_gray = self.to_gray(atlas.preprocessed_image)
            score = structural_similarity(target_image_gray, atlas_gray, data_range=1)
            results.append(AtlasScore(atlas, score))
        results.sort(key=lambda x: x.score, reverse=True)
        return results

