import argparse
import os
import sys
import json
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict
from torchvision.ops import nms
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# 경로 설정
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# SAM
from segment_anything import sam_model_registry, sam_hq_model_registry, SamPredictor


def load_image(image_path):
    # 원본 PIL 이미지
    orig_pil = Image.open(image_path).convert("RGB")
    # DINO 입력용 리사이즈
    resize = T.RandomResize([800], max_size=1333)
    dino_pil, _ = resize(orig_pil, None)
    # 텐서 변환 및 정규화
    to_tensor = T.ToTensor()
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    image_tensor, _ = to_tensor(dino_pil, None)
    image_tensor, _ = normalize(image_tensor, None)
    return orig_pil, dino_pil, image_tensor


def load_model(model_config_path, model_checkpoint_path, bert_base_uncased_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    args.bert_base_uncased_path = bert_base_uncased_path
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."

    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes = outputs["pred_boxes"].cpu()[0]

    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]
    boxes_filt = boxes[filt_mask]

    # tokenlizer = model.tokenizer
    # tokenized = tokenlizer(caption)

    scores = logits_filt.max(dim=1).values  # torch.Tensor of shape [M]
    tokenlizer = model.tokenizer  
    tokenized = tokenlizer(caption)

    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases, scores


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list):
    if len(mask_list) == 0:
        return

    first_mask = torch.tensor(mask_list[0])
    if first_mask.ndim == 3:
        first_mask = first_mask.squeeze(0)
    mask_img = torch.zeros_like(first_mask, dtype=torch.int32)
    for idx, mask in enumerate(mask_list):
        mask_tensor = torch.tensor(mask)
        if mask_tensor.ndim == 3:
            mask_tensor = mask_tensor.squeeze(0)
        mask_img[mask_tensor.bool()] = idx + 1

    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{'value': 0, 'label': 'background'}]
    for idx, (label, box) in enumerate(zip(label_list, box_list)):
        name, logit = label.split('(')
        logit = float(logit[:-1])
        json_data.append({
            'value': idx+1,
            'label': name.strip(),
            'logit': logit,
            'box': [float(b) for b in box]
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--grounded_checkpoint", type=str, required=True)
    parser.add_argument("--sam_version", type=str, default="vit_h")
    parser.add_argument("--sam_checkpoint", type=str, required=False)
    parser.add_argument("--sam_hq_checkpoint", type=str, default=None)
    parser.add_argument("--use_sam_hq", action="store_true")
    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--text_prompt", type=str, required=True)
    parser.add_argument("--output_dir", "-o", type=str, default="outputs", required=True)
    parser.add_argument("--box_threshold", type=float, default=0.3)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument('--nms_iou', type=float, default=0.95, help='IoU threshold for NMS')
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--bert_base_uncased_path", type=str, required=False)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 이미지 로드
    orig_pil, dino_pil, image = load_image(args.input_image)
    orig_W, orig_H = orig_pil.size
    dino_W, dino_H = dino_pil.size

    # 모델 로드 및 원본 이미지 저장
    model = load_model(args.config, args.grounded_checkpoint, args.bert_base_uncased_path, device=args.device)
    orig_pil.save(os.path.join(args.output_dir, "raw_image.jpg"))

    # grounding 출력
    boxes_norm, pred_phrases,scores = get_grounding_output(
        model, image, args.text_prompt, args.box_threshold, args.text_threshold, device=args.device
    )

    # 박스 정규화 -> 픽셀 단위 (DINO 크기)
    boxes_dino = boxes_norm * torch.tensor([dino_W, dino_H, dino_W, dino_H])
    boxes_dino[:, :2] -= boxes_dino[:, 2:] / 2
    boxes_dino[:, 2:] += boxes_dino[:, :2]
    scale_x = orig_W / dino_W
    scale_y = orig_H / dino_H
    boxes_orig = boxes_dino.clone()
    boxes_orig[:, [0, 2]] *= scale_x
    boxes_orig[:, [1, 3]] *= scale_y

    keep = nms(boxes_orig, scores, args.nms_iou)
    boxes_orig = boxes_orig[keep]
    pred_phrases = [pred_phrases[i] for i in keep.tolist()]

    # SAM 초기화
    if args.use_sam_hq:
        predictor = SamPredictor(sam_hq_model_registry[args.sam_version](checkpoint=args.sam_hq_checkpoint).to(args.device))
    else:
        predictor = SamPredictor(sam_model_registry[args.sam_version](checkpoint=args.sam_checkpoint).to(args.device))

    cv_image = cv2.imread(args.input_image)
    image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    # 예측 및 마스크 저장
    label_to_masks = defaultdict(list)
    all_masks = []
    boxes_cpu = boxes_orig.cpu()
    for box, label in zip(boxes_cpu, pred_phrases):
        transformed = predictor.transform.apply_boxes_torch(box.unsqueeze(0), image_rgb.shape[:2]).to(args.device)
        mask, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed,
            multimask_output=False,
        )
        clean_lbl = label.split('(')[0].strip().lower()
        label_to_masks[clean_lbl].append(mask[0].cpu().numpy())

    # 개별 라벨 마스크 이미지 저장
    for lbl, masks in label_to_masks.items():
        combined = np.any(np.stack(masks, axis=0), axis=0).astype(np.uint8) * 255
        combined = np.squeeze(combined)  # (1,H,W) -> (H,W)
        Image.fromarray(combined).save(os.path.join(args.output_dir, f"{lbl}_mask.png"))
        all_masks.extend(masks)

    # 결과 시각화
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    for m in all_masks:
        show_mask(m, plt.gca(), random_color=True)
    for b, l in zip(boxes_cpu, pred_phrases):
        show_box(b.numpy(), plt.gca(), l)
    plt.axis('off')
    plt.savefig(os.path.join(args.output_dir, "grounded_sam_output.jpg"), bbox_inches="tight", dpi=300, pad_inches=0.0)

    # 마스크 JSON 및 이미지 저장
    save_mask_data(args.output_dir, all_masks, boxes_cpu, pred_phrases)
