import os
import json
import subprocess
import time
from PIL import Image
import trimesh
import argparse

def run_pipeline(image_path, prompt, session_name):
    start = time.time()

    base_dir = "/home/ldy/demo"
    input_dir = os.path.join(base_dir, "input", session_name)
    output_dir = os.path.join(base_dir, "output", session_name)
    preprocessed_path = os.path.join(output_dir, "preprocessed.png")
    painted_path = os.path.join(output_dir, "painted.png")
    mask_output_dir = os.path.join(output_dir, "masks")

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(mask_output_dir, exist_ok=True)

    # 입력 저장
    input_img_path = os.path.join(input_dir, "original.jpg")
    input_prompt_path = os.path.join(input_dir, "prompt.txt")
    img = Image.open(image_path)
    img.save(input_img_path)
    with open(input_prompt_path, "w") as f:
        f.write(prompt)

    # 1. Prompt → JSON
    parsed_json_path = os.path.join(output_dir, "parsed.json")
    subprocess.run(
        f"python parsing.py --prompt \"{prompt}\" --output {parsed_json_path}",
        shell=True, cwd=base_dir, check=True
    )

    with open(parsed_json_path) as f:
        parsed = json.load(f)

    object_name = parsed["object"]
    part_prompts = [part["part_name"] for part in parsed["parts"] if part["part_name"] != object_name]
    part_prompts.append(object_name)

    # 2. 전처리
    subprocess.run(
        f"conda run -n GroundedSam python preprocess.py --input {input_img_path} --output {preprocessed_path}",
        shell=True, cwd=base_dir, check=True
    )

    # 3. GroundedSAM
    for part in part_prompts[:-1]:
        text_prompt = f"{part}, {object_name}"
        subprocess.run(
            f"conda run -n GroundedSam python grounded_sam.py "
            f"--config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py "
            f"--grounded_checkpoint groundingdino_swint_ogc.pth "
            f"--sam_checkpoint sam_vit_h_4b8939.pth "
            f"--input_image {preprocessed_path} --output_dir {mask_output_dir} "
            f"--box_threshold 0.25 --text_threshold 0.25 "
            f"--text_prompt \"{text_prompt}\" --device cuda",
            shell=True, cwd=os.path.join(base_dir, "GroundedSAM"), check=True
        )

    # 4. ControlNet
    controlnet_prompt_path = os.path.join(output_dir, "controlnet_prompts.json")
    with open(controlnet_prompt_path) as f:
        controlnet_prompts = json.load(f)

    src_path = input_img_path
    for entry in controlnet_prompts:
        part_name = entry["part_name"]
        prompt_text = entry["prompt"]
        mask_path = os.path.join(mask_output_dir, f"{part_name}_mask.png")

        ret = subprocess.run(
            f"conda run -n control python inpaint.py \"{src_path}\" \"{mask_path}\" \"{painted_path}\" \"{prompt_text}\"",
            shell=True, cwd=os.path.join(base_dir, "ControlNet")
        )
        if ret.returncode == 0:
            src_path = painted_path

    # 5. InstantMesh
    subprocess.run(
        f"conda run -n instantmesh python run.py configs/instant-mesh-base.yaml {painted_path} --export_texmap --save_video",
        shell=True, cwd=os.path.join(base_dir, "InstantMesh"), check=True
    )

    final_video_path = os.path.join(base_dir, "InstantMesh", "outputs", "instant-mesh-base", "videos", "painted.mp4")
    mesh_path = os.path.join(base_dir, "InstantMesh", "outputs", "instant-mesh-base", "meshes", "painted.obj")

    if not os.path.exists(final_video_path):
        raise FileNotFoundError(f"[InstantMesh] 결과 영상이 없음: {final_video_path}")

    # 6. GLB
    mesh = trimesh.load(mesh_path, force='mesh', skip_materials=False)
    glb_path = os.path.join(output_dir, "result.glb")
    mesh.export(glb_path)

    print(f"[완료] 실행 시간: {time.time() - start:.2f}초")
    print(f"[결과] 영상: {final_video_path}")
    print(f"[결과] GLB: {glb_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="입력 이미지 경로")
    parser.add_argument("--prompt", required=True, help="프롬프트")
    parser.add_argument("--session", required=True, help="세션 이름")
    args = parser.parse_args()

    run_pipeline(args.image, args.prompt, args.session)
