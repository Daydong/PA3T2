import os
import json
import subprocess
import gradio as gr
from PIL import Image
import trimesh
import time


def run_pipeline(image, prompt, session_name):
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
    image.save(input_img_path)
    with open(input_prompt_path, "w") as f:
        f.write(prompt)

    # 1. Prompt -> JSON 파싱
    parsed_json_path = os.path.join(output_dir, "parsed.json")
    ret = subprocess.run(
        f"python parsing.py --prompt \"{prompt}\" --output {parsed_json_path}",
        shell=True, cwd=base_dir
    )
    if ret.returncode != 0:
        raise RuntimeError("[Error] Failed at parsing stage.")

    with open(parsed_json_path) as f:
        parsed = json.load(f)
    object_name = parsed["object"]
    part_prompts = [part["part_name"] for part in parsed["parts"] if part["part_name"] != object_name]
    part_prompts.append(object_name)

    # 2. 전처리
    ret = subprocess.run(
        f"conda run -n GroundedSam python preprocess.py --input {input_img_path} --output {preprocessed_path}",
        shell=True, cwd=base_dir
    )
    if ret.returncode != 0:
        raise RuntimeError("[Error] preprocess.py failed.")

    # 3. GroundedSAM
    object_name = part_prompts[-1]
    parts = part_prompts[:-1]
    for part in parts:
        text_prompt = f"{part}, {object_name}"
        print("[DEBUG] GroundedSAM 실행 인자:")
        print(f"--text_prompt \"{text_prompt}\"")

        ret = subprocess.run(
            f"conda run -n GroundedSam python grounded_sam_demo.py "
            f"--config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py "
            f"--grounded_checkpoint groundingdino_swint_ogc.pth "
            f"--sam_checkpoint sam_vit_h_4b8939.pth "
            f"--input_image {preprocessed_path} --output_dir {mask_output_dir} "
            f"--box_threshold 0.25 --text_threshold 0.25 "
            f"--text_prompt \"{text_prompt}\" --device cuda",
            shell=True, cwd=os.path.join(base_dir, "GroundedSAM")
        )
        if ret.returncode != 0:
            raise RuntimeError(f"[Error] GroundedSAM failed on part {part}.")

    # 4. ControlNet
    controlnet_prompt_path = os.path.join(output_dir, "controlnet_prompts.json")
    with open(controlnet_prompt_path, "r") as f:
        controlnet_prompts = json.load(f)

    src_path = input_img_path
    for entry in controlnet_prompts:
        part_name = entry["part_name"]
        prompt_text = entry["prompt"]
        mask_path = os.path.join(mask_output_dir, f"{part_name}_mask.png")

        ret = subprocess.run(
            f"conda run -n control python test.py \"{src_path}\" \"{mask_path}\" \"{painted_path}\" \"{prompt_text}\"",
            shell=True, cwd=os.path.join(base_dir, "ControlNet")
        )
        if ret.returncode != 0:
            print(f"[Warn] Inpainting failed: {part_name}")
        else:
            src_path = painted_path

    # 5. InstantMesh
    ret = subprocess.run(
        f"conda run -n instantmesh python run.py configs/instant-mesh-base.yaml {painted_path} --export_texmap --save_video",
        shell=True, cwd=os.path.join(base_dir, "InstantMesh")
    )

    final_video_path = os.path.join(base_dir, "InstantMesh", "outputs", "instant-mesh-base", "videos", "painted.mp4")
    mesh_path = os.path.join(base_dir, "InstantMesh", "outputs", "instant-mesh-base", "meshes", "painted.obj")

    if not os.path.exists(final_video_path):
        raise FileNotFoundError(f"[InstantMesh] 결과 영상이 생성되지 않았습니다: {final_video_path}")

    elapsed = time.time() - start 
    print(f"⏱️ 파이프라인 실행 시간: {elapsed:.2f}초")
    
    # 6. GLB 변환
    mesh = trimesh.load(mesh_path, force='mesh', skip_materials=False)
    glb_path = os.path.join(output_dir, "result.glb")
    mesh.export(glb_path)

    target_part = parsed["parts"][0]["part_name"]
    mask_img_path = os.path.join(mask_output_dir, f"{target_part}_mask.png")

    return [image], [Image.open(mask_img_path)], [Image.open(painted_path)], final_video_path, glb_path

# Gradio 인터페이스

demo = gr.Interface(
    fn=run_pipeline,
    inputs=[
        gr.Image(label="이미지 업로드", type="pil"),
        gr.Textbox(label="텍스처 스타일 프롬프트", placeholder="객체+스타일+부위 형식의 프롬프트"),
        gr.Textbox(label="세션 이름 (예: p1)", placeholder="p1")
    ],
    outputs=[
        gr.Gallery(label="입력 이미지", columns=1),
        gr.Gallery(label="분할 마스크 결과", columns=1),
        gr.Gallery(label="텍스처 변환 결과", columns=1),
        gr.Video(label="3D 영상 결과"),
        gr.Model3D(label="3D Mesh (glb)")
    ],
    title="DEMO - Single View",
    allow_flagging="never"
)

demo.queue().launch(share=True)
