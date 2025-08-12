from dataclasses import dataclass

from atlas.atlas import Atlas


@dataclass(frozen=True)
class AtlasScore:
    atlas: Atlas
    score: float