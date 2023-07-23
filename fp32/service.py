from hashlib import sha256
from instance import Instance

from contextlib import ExitStack
from starlette.middleware.cors import CORSMiddleware

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, DiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusionInpaintPipeline
from huggingface_hub import login
login(Instance.HUGGING_FACE_PWD)
from pydantic import BaseModel
import bentoml
from bentoml.io import Image, JSON, Multipart, NumpyNdarray
import time
import numpy as np
import boto3
import os
from diffusers.utils import pt_to_pil
import gc
from diffusers.utils import load_image
import logging

# torch.autocast(device_type="cpu", enabled=False)

class StableDiffusionRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):

        ##### model stable-diffusion-xl-base-0.9
        self.txt2img_pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-0.9", torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to("cuda")

        self.txt2img_refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-0.9", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        self.txt2img_refiner.enable_model_cpu_offload()
        self.img2img_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-0.9", torch_dtype=torch.float16)
        self.img2img_pipe.enable_model_cpu_offload()

    @bentoml.Runnable.method(batchable=False, batch_dim=0)
    def txt2img(self, data):

        id_ = data["id"]
        prompt = data["prompt"]
        guidance_scale = data.get('guidance_scale', 7.5)
        height = data.get('height', 1024)
        width = data.get('width', 1024)
        no_images = data.get('no_images', 2)
        num_inference_steps = data.get('num_inference_steps', 70)
        generator = torch.Generator('cuda')
        generator.manual_seed(data.get('seed'))

        gc.collect()
        torch.cuda.empty_cache()

        images_output = []
        images = self.txt2img_pipe(
                prompt=prompt,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                generator=generator,
                num_images_per_prompt=no_images,
                output_type="latent"
            ).images
        
        logging.error(f"images: {type(images)}")
        # logging.error(f"txt2img_refiner: {type(self.txt2img_refiner)}")
        
        for image in images:
            image_refiner = self.txt2img_refiner(prompt=prompt, image=image[None, :], guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, strength=0.9).images[0]
            images_output.append(image_refiner)

        images_output += images
        return images_output, prompt

    @bentoml.Runnable.method(batchable=False, batch_dim=0)
    def img2img(self, data):
        
        id_ = data["id"]
        prompt = data["prompt"]
        strength = data.get('strength', 0.8)
        guidance_scale = data.get('guidance_scale', 7.5)
        num_inference_steps = data.get('num_inference_steps', 50)
        generator = torch.Generator('cuda')
        generator.manual_seed(data.get('seed'))
        no_images = data.get('no_images', 2)
        image_url = data.get('image_url', '')

        gc.collect()
        torch.cuda.empty_cache()
        
        
        init_image = load_image(image_url).convert("RGB")
        
        images = self.img2img_pipe(
            prompt=prompt,
            image=init_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            num_images_per_prompt=no_images
        ).images
        
        return images, prompt


stable_diffusion_runner = bentoml.Runner(StableDiffusionRunnable, name='stable_diffusion_runner', max_batch_size=10)

svc = bentoml.Service("stable_diffusion_fp32", runners=[stable_diffusion_runner])
svc.add_asgi_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], expose_headers=["*"])


def generate_seed_if_needed(seed):
    if seed is None:
        generator = torch.Generator()
        seed = torch.seed()
    return seed

def upload_images(prompt, images):
    url = []
    

    bucket_name = Instance.S3_BUCKET_NAME


    # Upload the file
    s3_client = boto3.client('s3',
                            aws_access_key_id=Instance.AWS_ACCESS_KEY_ID,
                            aws_secret_access_key=Instance.AWS_SECRET_ACCESS_KEY)
    try:
        for image in images:
            file_name = f"{sha256(prompt.encode('utf-8')).hexdigest()}_{int(time.time()*1000000)}.jpeg"
            image.save(file_name)

            response = s3_client.upload_file(file_name, bucket_name, file_name)
            url.append(f"{Instance.S3_BUCKET_BASED_END_POINT}/{bucket_name}/{file_name}")
    except Exception as e:
        logging.error(e)
        return []
    return url

class Txt2ImgInput(BaseModel):
    id: int
    prompt: str
    guidance_scale: float = 7.5
    height: int = 512
    width: int = 512
    num_inference_steps: int = 50
    safety_check: bool = True
    seed: int = None
    no_images: int = 2

class Output(BaseModel):
    id: int
    images: list

@svc.api(input=JSON(pydantic_model=Txt2ImgInput), output=JSON())
def txt2img(data, context):
    data = data.dict()
    data['seed'] = generate_seed_if_needed(data['seed'])
    images, prompt = stable_diffusion_runner.txt2img.run(data)
    url = upload_images(prompt,images)
        
    for i in data:
        context.response.headers.append(i, str(data[i]))
        
    return {'image': url}

class Img2ImgInput(BaseModel):
    id: int
    prompt: str
    strength: float = 0.8
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    safety_check: bool = True
    seed: int = None
    image_url: str
    no_images: int = 2


@svc.api(input=JSON(pydantic_model=Img2ImgInput), output=JSON())
def img2img(data, context):
    data = data.dict()
    data['seed'] = generate_seed_if_needed(data['seed'])
    images, prompt = stable_diffusion_runner.img2img.run(data)
    url = upload_images(prompt,images)
        
    for i in data:
        context.response.headers.append(i, str(data[i]))
    return  {'image': url}

