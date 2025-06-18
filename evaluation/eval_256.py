import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from skimage.color import rgb2lab
from skimage.feature import graycomatrix, graycoprops
from colormath.color_diff import delta_e_cie2000 as delta_e_ciede2000
from colormath.color_objects import LabColor
from lpips import LPIPS

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA available:", torch.cuda.is_available())
print("Device:", device)

# LPIPS 모델
lpips_model = LPIPS(net='alex').to(device)

def read_video_frames(path, size=(256, 256), max_frames=30):
    cap = cv2.VideoCapture(path)
    frames = []
    tensors = []
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, size)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
        tensor = (
            torch.tensor(rgb / 127.5 - 1.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(device)
        )
        tensors.append(tensor)
    cap.release()
    return frames, tensors

def delta_e(frame1, frame2):
    lab1 = rgb2lab(frame1)
    lab2 = rgb2lab(frame2)
    diff = np.zeros(lab1.shape[:2])
    for i in range(lab1.shape[0]):
        for j in range(lab1.shape[1]):
            c1 = LabColor(*lab1[i, j])
            c2 = LabColor(*lab2[i, j])
            diff[i, j] = delta_e_ciede2000(c1, c2)
    return np.mean(diff)

def histogram_similarity(f1, f2):
    sim = 0
    for c in range(3):
        hist1 = cv2.calcHist([f1], [c], None, [256], [0, 256])
        hist2 = cv2.calcHist([f2], [c], None, [256], [0, 256])
        sim += cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return sim / 3

def glcm_contrast_diff(f1, f2):
    gray1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(f2, cv2.COLOR_RGB2GRAY)
    glcm1 = graycomatrix(gray1, [1], [0], 256, symmetric=True, normed=True)
    glcm2 = graycomatrix(gray2, [1], [0], 256, symmetric=True, normed=True)
    c1 = graycoprops(glcm1, 'contrast')[0, 0]
    c2 = graycoprops(glcm2, 'contrast')[0, 0]
    return abs(c1 - c2)

def evaluate_pipeline(ref_path, target_path):
    ref_frames, ref_tensors = read_video_frames(ref_path)
    tgt_frames, tgt_tensors = read_video_frames(target_path)
    n = min(len(ref_frames), len(tgt_frames))

    hist_sims = []
    deltaEs = []
    contrasts = []
    lpips_vals = []

    for i in range(n):
        f1 = ref_frames[i]
        f2 = tgt_frames[i]
        hist_sims.append(histogram_similarity(f1, f2))
        deltaEs.append(delta_e(f1, f2))
        contrasts.append(glcm_contrast_diff(f1, f2))

        print(i)


        with torch.no_grad():
            dist = lpips_model(ref_tensors[i], tgt_tensors[i]).item()
            lpips_vals.append(dist)

    return {
        "Histogram Similarity": np.mean(hist_sims),
        "Delta E": np.mean(deltaEs),
        "GLCM Contrast Diff": np.mean(contrasts),
        "LPIPS": np.mean(lpips_vals)
    }

def main():
    base_dir = os.path.join(os.getcwd(), "video", "l2")
    ref_path = os.path.join(base_dir, "std.mp4")
    pipelines = ["ours", "p3", "txt", "sm"]
    results = {}

    for name in pipelines:
        target_path = os.path.join(base_dir, f"{name}.mp4")
        metrics = evaluate_pipeline(ref_path, target_path)
        results[name] = metrics

    csv_path = "result.csv"
    df_new = pd.DataFrame(results).T
    df_new.reset_index(inplace=True)
    df_new.rename(columns={"index": "Pipeline"}, inplace=True)

    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(csv_path, index=False)
    print("✅ Saved to result.csv")

if __name__ == "__main__":
    main()
