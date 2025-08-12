from skimage.metrics import structural_similarity

from atlas.atlas_score import _to_gray, AtlasScore


class SimilarityAtlasSelector:
    def select_atlases(self, atlases, target_image, n):
        scored_atlases = self.score_atlases(atlases, target_image)
        return scored_atlases[:n]


    @staticmethod
    def score_atlases(atlases, target_image):
        target_image_gray = _to_gray(target_image)
        results = []
        for atlas in atlases:
            atlas_gray = _to_gray(atlas.preprocessed_image)
            score = structural_similarity(target_image_gray, atlas_gray, data_range=1)
            results.append(AtlasScore(atlas, score))
        results.sort(key=lambda x: x.score, reverse=True)
        return results