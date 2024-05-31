# Stability AI API

A small API wrapper to interact with the Stability AI API.

This is a work in progress and currently only the
[Stability AI API version 1](https://platform.stability.ai/docs/api-reference#tag/Text-to-Image)
is implemented, including the following services:

* Text-to-Image (models: SDXL v1.0, SD v1.6, SD Beta)
* Image-to-Image (models: SDXL v1.0, SD v1.6, SD Beta)
  * with prompt
  * with a mask
* Image-to-Image upscale (model: ESRGAN x2 Upscaler)

## Installation

```bash
python3 -m pip install stability-ai-api
```

## Usage

```python
from stability_ai_api.basic_types import EngineIdV1, StylePreset
from stability_ai_api.stability_ai_api import StabilityAiV1Solver
solver = StabilityAiV1Solver(api_key="your-stability-ai-api-key", engine_id=EngineIdV1.SDXL_10)
prompt = {"text": "a dog with funny hat holding a baseball bat"}
image = solver.tti_query(  # text-to-image
    prompts=prompt,
    width=1216,
    height=832,
    style_preset=StylePreset.FANTASY_ART
)
with open("example.png", "wb") as f:
    f.write(image)
```

## Notes

Please verify you have credits before using this API. The price (credits) per model generation
can be found at [Stability AI Pricing page](https://platform.stability.ai/pricing).
