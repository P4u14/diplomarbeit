from dataclasses import dataclass

from segmenter.atlas.atlas import Atlas


@dataclass(frozen=True)
class AtlasScore:
    """
    Data class representing the association of an Atlas object with a score value.
    Used to rank or evaluate atlases based on a computed score (e.g., similarity, relevance).

    Attributes:
        atlas (Atlas): The atlas object being scored.
        score (float): The score assigned to the atlas.
    """
    atlas: Atlas
    score: float