import os
import torch
import numpy as np
from PIL import Image
import argparse
from rembg import remove, new_session

# 인자 파싱
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True, help="Input image path")
parser.add_argument("--output", type=str, required=True, help="Directory to save processed image")
args = parser.parse_args()

# 경로 설정
image_path = args.input
output_path = args.output
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 전처리
image = Image.open(image_path).convert("RGBA")
session = new_session(model_name="isnet-general-use")
result = remove(image, session=session).convert("RGB")
upscaled = result.resize((1024, 1024), Image.BICUBIC)

# 저장
upscaled.save(output_path)
print(f"Preprocessed saved to {output_path}")
