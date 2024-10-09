import io
from time import sleep

import torch
from diffusers import StableDiffusion3Pipeline, StableDiffusionPipeline
from IModel import Model
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from openai import BadRequestError
from PIL import Image


class DALL_E(Model):
    def __init__(self, model_id="dall-e-3"):
        self.model = DallEAPIWrapper(model=model_id)
        self.model_name = model_id.split("/")[-1]

    def invoke(self, prompt):
        err = None
        for t in range(3):
            try:
                url = self.model.run(prompt)
                break
            except BadRequestError as e:
                err = e
                print(f"BadRequestError! retrying... ({t+1}/3)")
                sleep(1)
        else:
            raise err

        im = io.imread(url)
        im = Image.fromarray(im)
        return im


class SD(Model):
    def __init__(self, model_id="stabilityai/stable-diffusion-2"):
        model_id = model_id
        self.model_name = model_id.split("/")[-1]

        # SD3 uses a different class than previous versions
        if model_id.startswith("stabilityai/stable-diffusion-3"):
            self.pipe = StableDiffusion3Pipeline.from_pretrained(
                model_id, torch_dtype=torch.float16
            ).to("cuda")
        else:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id, torch_dtype=torch.float16
            ).to("cuda")

        self.pipe.set_progress_bar_config(disable=True)

    def invoke(self, prompt):
        image = self.pipe(prompt=prompt).images[0]
        return image


class SD3(SD):
    def __init__(self, model_id="stabilityai/stable-diffusion-3-medium-diffusers"):
        super().__init__(model_id=model_id)
        self.pipe.set_progress_bar_config(disable=True)
