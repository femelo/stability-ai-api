import sys
from dataclasses import dataclass
from typing import Union, Optional

STABILITY_AI_API_V1_URL_TEMPLATE = (
    "https://api.stability.ai/v1/generation/"
    "{engine_id}/{query_type}"
)
AUTHORIZATION_TEMPLATE = "Bearer {api_key}"
MAX_DIM = sys.maxsize

SDXL_ALLOWED_DIMENSIONS = [
    (1024, 1024),
    (1152,  896),
    ( 896, 1152),
    (1216,  832),
    (1344,  768),
    ( 768, 1344),
    (1536,  640),
    ( 640, 1536)
]
SD_16_DIMENSION_BOUNDS = (320, 1536)
SD_BETA_DIMENSION_BOUNDS = (128, 896)
SD_BETA_DIMENSION_BOUND = 512


class Default:
    height: int = 512
    width: int = 512
    cfg_scale: int = 7
    clip_guidance_preset: str = "NONE"
    sampler: Optional[str] = None
    samples: int = 1
    seed: int = 0
    steps: int = 30
    image_strength: float = 0.35
    step_schedule_start: float = 0.65
    step_schedule_end: float = 1.00
    mask_source: str = "MASK_IMAGE_WHITE"


@dataclass
class Interval:
    inf: Union[int, float] = 0
    sup: Union[int, float] = MAX_DIM


class ValidRange:
    height: Interval = Interval(128)
    width: Interval = Interval(128)
    upscale_height: Interval = Interval(512)
    upscale_width: Interval = Interval(512)
    cfg_scale: Interval = Interval(0, 35)
    samples: Interval = Interval(1, 10)
    seed: Interval = Interval(0, 4294967295)
    steps: Interval = Interval(10, 50)
    image_strength: Interval = Interval(0.0, 1.0)
    step_schedule_start: Interval = Interval(0.0, 1.0)
    step_schedule_end: Interval = Interval(0.0, 1.0)
