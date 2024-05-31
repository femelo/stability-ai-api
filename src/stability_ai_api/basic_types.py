from enum import StrEnum, auto
from pydantic import BaseModel
from typing import Optional
from .constants import (
    SDXL_ALLOWED_DIMENSIONS,
    SD_16_DIMENSION_BOUNDS,
    SD_BETA_DIMENSION_BOUNDS,
    SD_BETA_DIMENSION_BOUND,
)


# Auxiliary class to attribute string values for enums
class AutoName(StrEnum):
    def _generate_next_value_(
            name: str,
            start: int,
            count: int,
            last_values: list
        ) -> str:
        return name


class QueryType(StrEnum):
    TEXT_TO_IMAGE = "text-to-image"
    IMAGE_TO_IMAGE = "image-to-image"


class EngineIdV1(StrEnum):
    SDXL_10 = "stable-diffusion-xl-1024-v1-0"
    SD_16 = "stable-diffusion-v1-6"
    SD_BETA = "stable-diffusion-xl-beta-v2-2-2"
    ESRGAN_1 = "esrgan-v1-x2plus"


class ClipGuidancePreset(AutoName):
    NONE = auto()
    FAST_BLUE = auto()
    FAST_GREEN = auto()
    SIMPLE = auto()
    SLOW = auto()
    SLOWER = auto()
    SLOWEST = auto()


class Sampler(AutoName):
    DDIM = auto()
    DDPM = auto()
    K_DPMPP_2M = auto()
    K_DPMPP_2S_ANCESTRAL = auto()
    K_DPM_2 = auto()
    K_DPM_2_ANCESTRAL = auto()
    K_EULER = auto()
    K_EULER_ANCESTRAL = auto()
    K_HEUN = auto()
    K_LMS = auto()


class StylePreset(AutoName):
    THREE_D_MODEL = "3d-model"
    ANALOG_FILM = "analog-film"
    ANIME = "anime"
    CINEMATIC = "cinematic"
    COMIC_BOOK = "comic-book"
    DIGITAL_ART = "digital-art"
    ENHANCE = "enhance"
    FANTASY_ART = "fantasy-art"
    ISOMETRIC = "isometric"
    LINE_ART = "line-art"
    LOW_POLY = "low-poly"
    MODELING_COMPOUND = "modeling-compound"
    NEON_PUNK = "neon-punk"
    ORIGAMI = "origami"
    PHOTOGRAPHIC = "photographic"
    PIXEL_ART = "pixel-art"
    TILE_TEXTURE = "tile-texture"


class InitImageMode(AutoName):
    IMAGE_STRENGTH = auto()
    STEP_SCHEDULE = auto()


class MaskSource(AutoName):
    MASK_IMAGE_WHITE = auto()
    MASK_IMAGE_BLACK = auto()
    INIT_IMAGE_ALPHA = auto()

class ContentType(StrEnum):
    APPLICATION_JSON = "application/json"
    IMAGE_PNG = "image/png"


class TextPrompt(BaseModel):
    text: str
    weight: Optional[float] = None


# Dimension validator
class DimensionValidator:
    @staticmethod
    def validate_and_raise(engine_id: EngineIdV1, height: int, width: int) -> None:
        if engine_id == EngineIdV1.SDXL_10:
            valid = (
                ((height, width) in SDXL_ALLOWED_DIMENSIONS) or
                ((width, height) in SDXL_ALLOWED_DIMENSIONS)
            )
        elif engine_id == EngineIdV1.SD_16:
            valid = (
                (SD_16_DIMENSION_BOUNDS[0] <= height <= SD_16_DIMENSION_BOUNDS[1]) and
                (SD_16_DIMENSION_BOUNDS[0] <= width  <= SD_16_DIMENSION_BOUNDS[1])
            )
        elif engine_id == EngineIdV1.SD_BETA:
            valid = (
                (SD_BETA_DIMENSION_BOUNDS[0] <= height <= SD_BETA_DIMENSION_BOUNDS[1]) and
                (SD_BETA_DIMENSION_BOUNDS[0] <= width  <= SD_BETA_DIMENSION_BOUNDS[1]) and
                ((height <= SD_BETA_DIMENSION_BOUND) or (width <= SD_BETA_DIMENSION_BOUND))
            )
        else:
            raise ValueError(f"invalid engine {engine_id}.")
        # Raise if not valid
        if not valid:
            raise ValueError(f"dimension ({height}, {width}) invalid for engine {engine_id}.")
