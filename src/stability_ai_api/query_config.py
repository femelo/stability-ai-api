from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
)
from typing import (
    Optional,
    Any,
    List,
    Dict,
)
from .constants import (
    ValidRange,
    Default,
)
from .basic_types import (
    ContentType,
    TextPrompt,
    ClipGuidancePreset,
    Sampler,
    StylePreset,
    InitImageMode,
    MaskSource,
)


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
