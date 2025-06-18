#!/usr/bin/env python3
#	runwayml/stable-diffusion-v1-5, 중립색 이미지-> inpaint후 합성, 1024
import torch
import os
import sys
from PIL import Image
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel

from PIL import Image, ImageChops

def preprocess_image_with_neutral_mask(image: Image.Image, mask: Image.Image, fill_color=(127, 127, 127)) -> Image.Image:

    # 중립색 배경 이미지 생성
    neutral = Image.new("RGB", image.size, fill_color)

    # 마스크 내부만 중립색으로 덮음 (마스크 바깥은 원본 유지)
    inv_mask = ImageChops.invert(mask.convert("L"))
    processed = Image.composite(neutral, image, inv_mask)
    
    return processed


def load_rgb_image(path):
    try:
        img = Image.open(path).convert("RGB")
        print(f"Loaded image {path}: {img.size}, mode={img.mode}")
        return img
    except Exception as e:
        print(f"Failed to load image: {e}")
        return None

def load_mask_image(path):
    if not os.path.exists(path):
        print(f"부위가 탐지되지 않음.")
        return None
    try:
        img = Image.open(path).convert("L")
        print(f"Loaded mask {path}: {img.size}, mode={img.mode}")
        return img
    except Exception as e:
        print(f"Failed to load mask: {e}")
        return None


def main():
    if len(sys.argv) != 5:
        print("Usage: python controlnet_inpaint.py <image_path> <mask_path> <output_path> <prompt>")
        sys.exit(1)


    image_path = sys.argv[1]
    mask_path = sys.argv[2]
    output_path = sys.argv[3]
    prompt = sys.argv[4]
    negative_prompt = (
    "bad proportions, malformed, deformed, disfigured, duplicate, error, "
    "blurry, low quality, lowres, jpeg artifacts, distorted, "
    "color shift, color spill, unwanted lighting, reflection, glare, "
    "original texture, original color")



    # 이미지, 마스크 불러오기
    image = load_rgb_image(image_path)
    mask = load_mask_image(mask_path)

    image=image.resize((1024, 1024), Image.BICUBIC)
    n_image = preprocess_image_with_neutral_mask(image, mask)

    if image is None or mask is None:
        print("Image or mask failed to load.")
        sys.exit(1)

    if image.size != mask.size:
        print(f"[WARN] Resizing mask from {mask.size} to {image.size}")
        mask = mask.resize(image.size)
        
    from PIL import ImageFilter

    # 약하게 확장 (5~9픽셀)
    mask = mask.filter(ImageFilter.MaxFilter(5))  # 홀수만 가능 (3, 5, 7, 9, ...)
    
    image = image.copy()
    mask = mask.copy()

    # ControlNet 모델 (inpaint 전용)
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_inpaint",
        torch_dtype=torch.float16
    )

    # Stable Diffusion + ControlNet Inpaint 파이프라인
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch.float16
    )
    pipe.load_lora_weights("./7sLumina3D.safetensors", adapter_name="lumina3d")
    pipe.set_adapters(["lumina3d"], adapter_weights=[1.0])
    pipe.fuse_lora()

    pipe.to("cuda")
    generator = torch.manual_seed(42)

    print(f"[DEBUG] image type: {type(image)}, mask type: {type(mask)}")


    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=n_image,
        mask_image=mask,
        control_image=n_image,
        generator=generator,
        num_inference_steps=70,
        guidance_scale=15.0,
        controlnet_conditioning_scale=0.5,
    ).images[0]
    
    final= Image.composite(result, image, mask)


    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final.save(output_path)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
