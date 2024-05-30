"""Main module."""
import os
import sys
from enum import StrEnum, auto
from pydantic import BaseModel, Field
from dataclasses import dataclass
from base64 import b64decode
from typing import Optional, Union, List, Dict, Any
import requests
from urllib import parse as urlparse

STABILITY_AI_API_V1_URL_TEMPLATE = "https://api.stability.ai//v1/generation/{engine_id}/{query_type}"
AUTHORIZATION_TEMPLATE = "Bearer {api_key}"

@dataclass
class Default:
    height: int = 512
    width: int = 512
    cfg_scale: int = 7
    clip_guidance_preset: str = "NONE"
    sampler: Optional[str] = None
    samples: int = 1
    seed: int = 0,
    steps: int = 30
    image_strength: float = 0.35
    step_schedule_start: float = 0.65
    step_schedule_end: float = 1.00
    mask_source: str = "MASK_IMAGE_WHITE"


@dataclass
class Interval:
    inf: Union[int, float] = 0
    sup: Union[int, float] = sys.maxsize


@dataclass
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


def hyphenize(field: str):
    return field.replace("_", "-")


class QueryType(StrEnum):
    TEXT_TO_IMAGE = "text-to-image"
    IMAGE_TO_IMAGE = "image-to-image"


class EngineIdV1(StrEnum):
    SDXL_10 = "stable-diffusion-xl-1024-v1-0"
    SD_16 = "stable-diffusion-v1-6"
    SD_BETA = "stable-diffusion-xl-beta-v2-2-2"


class ClipGuidancePreset(AutoName):
    NONE = auto()
    FAST_BLUE = auto()
    FAST_GREEN = auto()
    NONE = auto()
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
    class Config:
        alias_generator = hyphenize
    Content_Type: ContentType = Field(default=ContentType.APPLICATION_JSON)
    Accept: ContentType = Field(default=ContentType.APPLICATION_JSON)
    Organization: Optional[str] = None
    Stability_Client_ID: Optional[str] = None
    Stability_Client_Version: Optional[str] = None
    Authorization: str


class GeneralV1Config(BaseModel):
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
    init_image: bytes
    init_image_mode: InitImageMode = InitImageMode.IMAGE_STRENGTH
    image_strength: float = Field(
        ge=ValidRange.image_strength.inf,
        le=ValidRange.image_strength.sup,
        default=Default.image_strength
    )

class ImageToImageV1ConfigB(GeneralV1Config):
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
        engine_id: EngineIdV1,
        content_type: str = "application/json",
        accept: str = "application/json",
        organization: Optional[str] = None,
        client_id: Optional[str] = None,
        client_version: Optional[str] = None,
    ):
        self.engine_id = engine_id
        self.header = HeaderV1(
            Content_Type=content_type,
            Accept=accept,
            Organization=organization,
            Stability_Client_ID=client_id,
            Stability_Client_Version=client_version,
            Authorization=AUTHORIZATION_TEMPLATE.format(api_key=api_key),
        )

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
        if self.header.Content_Type == ContentType.APPLICATION_JSON:
            data = response.json()
            output = list(
                map(
                    lambda item: b64decode(item["image"]),
                    data["artifacts"] if isinstance(data["artifacts"], list)
                    else [data["artifacts"]]
                )
            )
        elif self.header.Content_Type == ContentType.APPLICATION_JSON:
            output = response.content
        else:
            pass
        return output

    def tti_query(
        self,
        prompts: List[Dict[str, str]],
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
        parameters = TextToImageV1Config(
            text_prompts=list([TextPrompt(**prompt) for prompt in prompts]),
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
            header=self.header.model_dump(),
            payload=parameters.model_dump(),
        )

        return self._extract_data(response)

    def iti_query_by_strength(
        self,
        prompts: List[Dict[str, str]],
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
        parameters = ImageToImageV1ConfigA(
            text_prompts=list([TextPrompt(**prompt) for prompt in prompts]),
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
            header=self.header.model_dump(),
            payload=parameters.model_dump(),
        )

        return self._extract_data(response)
    
    def iti_query_by_schedule(
        self,
        prompts: List[Dict[str, str]],
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
        parameters = ImageToImageV1ConfigB(
            text_prompts=list([TextPrompt(**prompt) for prompt in prompts]),
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
            header=self.header.model_dump(),
            payload=parameters.model_dump(),
        )

        return self._extract_data(response)
    
    def iti_query_with_mask(
        self,
        prompts: List[Dict[str, str]],
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
        parameters = ImageToImageV1ConfigC(
            text_prompts=list([TextPrompt(**prompt) for prompt in prompts]),
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
            header=self.header.model_dump(),
            payload=parameters.model_dump(),
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
                    engine_id="esrgan-v1-x2plus",
                    query_type=QueryType.IMAGE_TO_IMAGE
                ),
                "upscale"
            ),
            header=self.header.model_dump(),
            payload=parameters.model_dump(),
        )

        return self._extract_data(response)
