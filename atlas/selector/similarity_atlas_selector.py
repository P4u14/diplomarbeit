from skimage.metrics import structural_similarity
from typing_extensions import override

from atlas.atlas_score import AtlasScore
from atlas.selector.base_atlas_selector import BaseAtlasSelector


class SimilarityAtlasSelector(BaseAtlasSelector):
    @override
    def select_atlases(self, atlases, target_image, n):
        scored_atlases = self.score_atlases(atlases, target_image)
        return scored_atlases[:n]


    def score_atlases(self, atlases, target_image):
        target_image_gray = self.to_gray(target_image.preprocessed_image)
        results = []
        for atlas in atlases:
            atlas_gray = self.to_gray(atlas.preprocessed_image)
            score = structural_similarity(target_image_gray, atlas_gray, data_range=1)
            results.append(AtlasScore(atlas, score))
        results.sort(key=lambda x: x.score, reverse=True)
        return results