"""Main module."""
import os
import sys
from enum import StrEnum, auto
from pydantic import BaseModel, ConfigDict, Field
from dataclasses import dataclass
from base64 import b64decode
from typing import Optional, Union, List, Dict, Any
import requests
from urllib import parse as urlparse

STABILITY_AI_API_V1_URL_TEMPLATE = "https://api.stability.ai/v1/generation/{engine_id}/{query_type}"
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


class AutoName(StrEnum):
    def _generate_next_value_(name: str, start: int, count: int, last_values: list) -> str:
        return name


class QueryType(StrEnum):
    TEXT_TO_IMAGE = "text-to-image"
    IMAGE_TO_IMAGE = "image-to-image"


class EngineIdV1(StrEnum):
    SDXL_10 = "stable-diffusion-xl-1024-v1-0"
    SD_16 = "stable-diffusion-v1-6"
    SD_BETA = "stable-diffusion-xl-beta-v2-2-2"
    ESRGAN_1 = "esrgan-v1-x2plus"


class DimensionValidator:
    @staticmethod
    def validate_and_raise(engine_id: EngineIdV1, height: int, width: int) -> None:
        if engine_id == EngineIdV1.SDXL_10:
            valid = (height, width) in SDXL_ALLOWED_DIMENSIONS
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
            raise ValueError("invalid engine {engine_id}.")

        if not valid:
            raise ValueError("dimension ({height}, {width}) invalid for engine {engine_id}.")


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


class HeaderV1(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    content_type: ContentType = Field(alias="Content-Type", default=ContentType.APPLICATION_JSON)
    accept: ContentType = Field(alias="Accept", default=ContentType.APPLICATION_JSON)
    organization: Optional[str] = Field(alias="Organization", default=None)
    stability_client_id: Optional[str] = Field(alias="Stability-Client-ID", default=None)
    stability_client_version: Optional[str] = Field(alias="Stability-Client-Version", default=None)
    authorization: str = Field(alias="Authorization")


class GeneralV1Config(BaseModel):
    # model_config = ConfigDict(use_enum_values=True)
    text_prompts: List[TextPrompt]
    cfg_scale: int = Field(
        ge=ValidRange.cfg_scale.inf,
        le=ValidRange.cfg_scale.sup,
        default=Default.cfg_scale
    )
    clip_guidance_preset: ClipGuidancePreset = Field(
        default=ClipGuidancePreset.NONE
    )
    sampler: Optional[Sampler] = None
    samples: int = Field(
        ge=ValidRange.samples.inf,
        le=ValidRange.samples.sup,
        default=Default.samples
    )
    seed: int = Field(
        ge=ValidRange.seed.inf,
        le=ValidRange.seed.sup,
        default=Default.seed
    )
    steps: int = Field(
        ge=ValidRange.steps.inf,
        le=ValidRange.steps.sup,
        default=Default.steps
    )
    style_preset: Optional[StylePreset] = None
    extras: Optional[Dict[str, Any]] = None


class TextToImageV1Config(GeneralV1Config):
    height: int = Field(
        multiple_of=64,
        ge=ValidRange.height.inf,
        default=Default.height
    )
    width: int = Field(
        multiple_of=64,
        ge=ValidRange.width.inf,
        default=Default.width
    )


class ImageToImageV1ConfigA(GeneralV1Config):
    # model_config = ConfigDict(use_enum_values=True)
    init_image: bytes
    init_image_mode: InitImageMode = InitImageMode.IMAGE_STRENGTH
    image_strength: float = Field(
        ge=ValidRange.image_strength.inf,
        le=ValidRange.image_strength.sup,
        default=Default.image_strength
    )

class ImageToImageV1ConfigB(GeneralV1Config):
    # model_config = ConfigDict(use_enum_values=True)
    init_image: bytes
    init_image_mode: InitImageMode = InitImageMode.STEP_SCHEDULE
    step_schedule_start: float = Field(
        ge=ValidRange.step_schedule_start.inf,
        le=ValidRange.step_schedule_start.sup,
        default=Default.step_schedule_start
    )
    step_schedule_end: float = Field(
        ge=ValidRange.step_schedule_end.inf,
        le=ValidRange.step_schedule_end.sup,
        default=Default.step_schedule_end
    )


class ImageToImageV1ConfigC(GeneralV1Config):
    # model_config = ConfigDict(use_enum_values=True)
    init_image: bytes
    mask_image: bytes
    mask_source: MaskSource


class ImageToImageV1ConfigD(BaseModel):
    image: bytes
    height: int = Field(
        multiple_of=64,
        ge=ValidRange.upscale_height.inf
    )
    width: int = Field(
        multiple_of=64,
        ge=ValidRange.upscale_width.inf
    )


class StabilityAiV1Solver:
    """Base class for solvers (API v1)."""
    def __init__(
        self,
        api_key: str,
        engine_id: EngineIdV1 = EngineIdV1.SDXL_10,
        content_type: str = "application/json",
        accept: str = "application/json",
        organization: Optional[str] = None,
        client_id: Optional[str] = None,
        client_version: Optional[str] = None,
    ):
        self.engine_id = engine_id
        try:
            self.header = HeaderV1(
                content_type=content_type,
                accept=accept,
                organization=organization,
                stability_client_id=client_id,
                stability_client_version=client_version,
                authorization=AUTHORIZATION_TEMPLATE.format(api_key=api_key),
            )
        except Exception as e:
            print(e)
            raise

    @staticmethod
    def _open_image(image: Union[bytes, str]) -> bytes:
        if isinstance(image, str):
            path = os.path.abspath(os.path.expanduser(image))
            if not os.path.exists(path):
                raise IOError("init image not found.")
            with open(path, "rb") as f:
                _image = f.read()
        else:
            _image = image
        return _image

    @staticmethod
    def _request(
        url: str,
        header: Dict[str, Any],
        payload: Dict[str, Any]
    ) -> requests.Response:
        """Helper to post the request and retrieve data"""
        try:
            response = requests.post(
                url=url,
                headers=header,
                json=payload,
            )
            response.raise_for_status()
        except requests.exceptions.HTTPError as error:
            raise SystemExit(error)
        return response

    def _extract_data(self, response: requests.Response) ->  Union[bytes, List[bytes]]:
        """Extract data from response."""
        if self.header.content_type == ContentType.APPLICATION_JSON:
            data = response.json()
            output = list(
                map(
                    lambda item: b64decode(item["base64"]),
                    data["artifacts"] if isinstance(data["artifacts"], list)
                    else [data["artifacts"]]
                )
            )
            if len(output) == 1:
                output = output[0]
        elif self.header.content_type == ContentType.IMAGE_PNG:
            output = response.content
        else:
            pass
        return output

    def tti_query(
        self,
        prompts: Union[Dict[str, str], List[Dict[str, str]]],
        height: int = Default.height,
        width: int = Default.width,
        cfg_scale: int = Default.cfg_scale,
        clip_guidance_preset: Optional[str] = Default.clip_guidance_preset,
        sampler:  Optional[str] = None,
        samples: int = Default.samples,
        seed: int = Default.seed,
        steps: int = Default.steps,
        style_preset: Optional[str] = None,
        extras: Optional[Dict[str, Any]] = None
    ) -> Union[bytes, List[bytes]]:
        """Perform text-to-image query."""
        DimensionValidator.validate_and_raise(self.engine_id, height, width)
        _prompts = prompts if isinstance(prompts, list) else [prompts]
        parameters = TextToImageV1Config(
            text_prompts=list([TextPrompt(**prompt) for prompt in _prompts]),
            height=height,
            width=width,
            cfg_scale=cfg_scale,
            clip_guidance_preset=clip_guidance_preset,
            sampler=sampler,
            samples=samples,
            seed=seed,
            steps=steps,
            style_preset=style_preset,
            extras=extras
        )

        response = StabilityAiV1Solver._request(
            url=STABILITY_AI_API_V1_URL_TEMPLATE.format(
                engine_id=self.engine_id,
                query_type=QueryType.TEXT_TO_IMAGE
            ),
            header=self.header.model_dump(exclude_none=True, by_alias=True, mode='json'),
            payload=parameters.model_dump(exclude_none=True, mode='json'),
        )

        return self._extract_data(response)

    def iti_query_by_strength(
        self,
        prompts: Union[Dict[str, str], List[Dict[str, str]]],
        init_image: Union[bytes, str],
        image_strength: float = Default.image_strength,
        cfg_scale: int = Default.cfg_scale,
        clip_guidance_preset: Optional[str] = Default.clip_guidance_preset,
        sampler:  Optional[str] = None,
        samples: int = Default.samples,
        seed: int = Default.seed,
        steps: int = Default.steps,
        style_preset: Optional[str] = None,
        extras: Optional[Dict[str, Any]] = None,
    ) -> Union[bytes, List[bytes]]:
        """Perform image-to-image query by image strength."""
        _init_image = StabilityAiV1Solver._open_image(init_image)
        _prompts = prompts if isinstance(prompts, list) else [prompts]
        parameters = ImageToImageV1ConfigA(
            text_prompts=list([TextPrompt(**prompt) for prompt in _prompts]),
            init_image=_init_image,
            image_strength=image_strength,
            cfg_scale=cfg_scale,
            clip_guidance_preset=clip_guidance_preset,
            sampler=sampler,
            samples=samples,
            seed=seed,
            steps=steps,
            style_preset=style_preset,
            extras=extras
        )

        response = StabilityAiV1Solver._request(
            url=STABILITY_AI_API_V1_URL_TEMPLATE.format(
                engine_id=self.engine_id,
                query_type=QueryType.IMAGE_TO_IMAGE
            ),
            header=self.header.model_dump(exclude_none=True, by_alias=True, mode='json'),
            payload=parameters.model_dump(exclude_none=True, mode='json'),
        )

        return self._extract_data(response)
    
    def iti_query_by_schedule(
        self,
        prompts: Union[Dict[str, str], List[Dict[str, str]]],
        init_image: Union[bytes, str],
        step_schedule_start: float = Default.step_schedule_start,
        step_schedule_end: float = Default.step_schedule_end,
        cfg_scale: int = Default.cfg_scale,
        clip_guidance_preset: Optional[str] = Default.clip_guidance_preset,
        sampler:  Optional[str] = None,
        samples: int = Default.samples,
        seed: int = Default.seed,
        steps: int = Default.steps,
        style_preset: Optional[str] = None,
        extras: Optional[Dict[str, Any]] = None,
    ) -> Union[bytes, List[bytes]]:
        """Perform image-to-image query by schedule step."""
        _init_image = StabilityAiV1Solver._open_image(init_image)
        _prompts = prompts if isinstance(prompts, list) else [prompts]
        parameters = ImageToImageV1ConfigB(
            text_prompts=list([TextPrompt(**prompt) for prompt in _prompts]),
            init_image=_init_image,
            step_schedule_start=step_schedule_start,
            step_schedule_end=step_schedule_end,
            cfg_scale=cfg_scale,
            clip_guidance_preset=clip_guidance_preset,
            sampler=sampler,
            samples=samples,
            seed=seed,
            steps=steps,
            style_preset=style_preset,
            extras=extras
        )

        response = StabilityAiV1Solver._request(
            url=STABILITY_AI_API_V1_URL_TEMPLATE.format(
                engine_id=self.engine_id,
                query_type=QueryType.IMAGE_TO_IMAGE
            ),
            header=self.header.model_dump(exclude_none=True, by_alias=True, mode='json'),
            payload=parameters.model_dump(exclude_none=True, mode='json'),
        )

        return self._extract_data(response)
    
    def iti_query_with_mask(
        self,
        prompts: Union[Dict[str, str], List[Dict[str, str]]],
        init_image: Union[bytes, str],
        mask_image: Union[bytes, str],
        mask_source: MaskSource = Default.mask_source,
        cfg_scale: int = Default.cfg_scale,
        clip_guidance_preset: Optional[str] = Default.clip_guidance_preset,
        sampler:  Optional[str] = None,
        samples: int = Default.samples,
        seed: int = Default.seed,
        steps: int = Default.steps,
        style_preset: Optional[str] = None,
        extras: Optional[Dict[str, Any]] = None,
    ) -> Union[bytes, List[bytes]]:
        """Perform image-to-image query with mask."""
        _init_image = StabilityAiV1Solver._open_image(init_image)
        _mask_image = StabilityAiV1Solver._open_image(mask_image)
        _prompts = prompts if isinstance(prompts, list) else [prompts]
        parameters = ImageToImageV1ConfigC(
            text_prompts=list([TextPrompt(**prompt) for prompt in _prompts]),
            init_image=_init_image,
            mask_image=_mask_image,
            mask_source=mask_source,
            cfg_scale=cfg_scale,
            clip_guidance_preset=clip_guidance_preset,
            sampler=sampler,
            samples=samples,
            seed=seed,
            steps=steps,
            style_preset=style_preset,
            extras=extras
        )

        response = StabilityAiV1Solver._request(
            url=urlparse.urljoin(
                STABILITY_AI_API_V1_URL_TEMPLATE.format(
                    engine_id=self.engine_id,
                    query_type=QueryType.IMAGE_TO_IMAGE
                ),
                "masking"
            ),
            header=self.header.model_dump(exclude_none=True, by_alias=True, mode='json'),
            payload=parameters.model_dump(exclude_none=True, mode='json'),
        )

        return self._extract_data(response)
    
    def iti_query_upscale(
        self,
        image: Union[bytes, str],
        height: int,
        width: int,
    ) -> Union[bytes, List[bytes]]:
        """Perform image-to-image query to upscale."""
        _image = StabilityAiV1Solver._open_image(image)
        parameters = ImageToImageV1ConfigD(
            image=_image,
            height=height,
            width=width,
        )

        response = StabilityAiV1Solver._request(
            url=urlparse.urljoin(
                STABILITY_AI_API_V1_URL_TEMPLATE.format(
                    engine_id=EngineIdV1.ESRGAN_1,
                    query_type=QueryType.IMAGE_TO_IMAGE
                ),
                "upscale"
            ),
            header=self.header.model_dump(exclude_none=True, by_alias=True, mode='json'),
            payload=parameters.model_dump(exclude_none=True, mode='json'),
        )

        return self._extract_data(response)


if __name__ == "__main__":
    solver = StabilityAiV1Solver(api_key="your-stability-ai-api-key")
    prompt = {"text": "a dog with funny hat holding a baseball bat"}
    image = solver.tti_query(
        prompts=prompt,
        width=1216,
        height=832,
        style_preset=StylePreset.FANTASY_ART
    )
    with open("example.png", "wb") as f:
        f.write(image)
