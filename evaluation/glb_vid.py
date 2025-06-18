import subprocess
import os

# 파이프라인 이름 목록
names = ["ours", "p3", "txt", "std", "sm"]

# 입력/출력 디렉토리
input_dir = "glb/l1"
output_dir = "video/l1"

for name in names:
    input_path = os.path.join(input_dir, f"{name}.glb")
    output_path = os.path.join(output_dir, f"{name}.mp4")
    cmd = ["python", "video.py", "-i", input_path, "-o", output_path]
    print(f"▶ Running: {' '.join(cmd)}")
    subprocess.run(cmd)
