"""Stability AI API v1 module."""
import os
import requests
from urllib import parse as urlparse
from base64 import b64decode
from typing import (
    Optional,
    Union,
    Any,
    List,
    Dict,
)
from .constants import (
    STABILITY_AI_API_V1_URL_TEMPLATE,
    AUTHORIZATION_TEMPLATE,
    Default,
)
from .basic_types import (
    EngineIdV1,
    QueryType,
    ContentType,
    StylePreset,
    TextPrompt,
    MaskSource,
    DimensionValidator,
)
from .query_config import (
    HeaderV1,
    TextToImageV1Config,
    ImageToImageV1ConfigA,
    ImageToImageV1ConfigB,
    ImageToImageV1ConfigC,
    ImageToImageV1ConfigD,
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
        self.header = HeaderV1(
            content_type=content_type,
            accept=accept,
            organization=organization,
            stability_client_id=client_id,
            stability_client_version=client_version,
            authorization=AUTHORIZATION_TEMPLATE.format(api_key=api_key),
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
