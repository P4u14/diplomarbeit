from dataclasses import dataclass

from segmenter.atlas.atlas import Atlas


@dataclass(frozen=True)
class AtlasScore:
    atlas: Atlas
    score: float