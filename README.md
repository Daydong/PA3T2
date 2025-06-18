# Text-Guided 3D Part Texture Editing 

![demo](./assets/demo.png)

사용자의 텍스트 요구사항을 기반으로, 해당 내용이 반영된 3D 결과물을 제공해주는 파이프라인입니다.  
Grounded-SAM, ControlNet, Instant Mesh 등을 통합하여 자동화된 처리 흐름을 제공합니다.

---



## 🧑‍💻 팀원 및 역할

| 이름   | GitHub                                      | 역할                                 |
|--------|---------------------------------------------|--------------------------------------|
| 이다예(리더)   | [@Daydong](https://github.com/Daydong) | 선행 연구 검토, 모듈 & 실험 설계 및 구현, 결과 분석 및 정리     |
| 이상은 | [@bingo4s](https://github.com/bingo4s)       | 선행 연구 검토, Baseline 구현 및 관리, 결과 분석 및 정리 |
| 정지우 | [@jiu-jung](https://github.com/jiu-jung)     | 선행 연구 검토, Baseline 구현 및 관리, 결과 분석 및 정리           |

---



## 📁 폴더 구조

```
.
├── assets/ # README 등에 사용되는 시각 자료 이미지
├── input/ # Gradio 실행 시 입력 이미지와 프롬프트 텍스트가 저장되는 폴더
├── output/ # 파이프라인 실행 후 생성된 결과물 (분석된 JSON, 보정 이미지, 3D 메쉬, 영상 등)이 저장됨
├── envs/ # 파이프라인에 필요한 conda 환경 정보(.yml 파일들) 저장 위치
├── ControlNet/ # ControlNet 관련 코드 및 모델 구성
├── GroundedSAM/ # Grounded-SAM 관련 코드 및 모델 구성
├── InstantMesh/ # Instant Mesh 변환 관련 코드
└── evaluation/ # 결과 평가 관련 모듈 (optional)
```



## 📄 구현 모듈 관련 파일 설명

### main
- `main.py`: 전체 파이프라인 실행 엔트리포인트
- `gradio_demo.py`: Gradio UI를 통해 파이프라인 실행을 원하는 경우 사용
- `parsing.py`: 프롬프트 파싱 및 가공 코드.
- `preprocess.py`: 이미지 업스케일 밎 전처리.

### GroundedSAM
- `grounded_sam.py`: 분할 모델을 이용한 분할 및 후처리 과정 모듈

### ControlNet
- `inpaint.py`: 이미지에 대한 inpainting 수행 모듈

### evaluation
- `eval.py`: 생성 결과물에 대한 metric 평가를 수행합니다. 'eval_256' 의 경우 더 빠른 평가를 위해 비디오 캡쳐 이미지 사이즈를 줄여 진행합니다.
- `glb_vid.py`: 생성된 glb에 대해 비디오를 생성합니다.
- `video.py`: 생성된 glb가 동일한 형태로 정렬되지 않은 경우 사용합니다. 단, scale이나 각도 등 정밀한 조정은 보장하지 않으므로, 직접 수치를 조정해야합니다.  

---



## ⚙️ 실행 환경 및 방법

- 저장소를 clone 하고, 해당 폴더로 이동합니다.

```
git clone https://github.com/Daydong/PA3T2
cd PA3T2
```

### 1. 환경 설정

- 기준 환경은, Ubuntu-20.04, RTX 4090 기준으로 구성되었습니다.
- 필요한 Conda 환경은 아래와 같습니다. 각 환경은 `/envs` 폴더에 있는 `.yml` 파일을 통해 설치할 수 있습니다.

```bash
cd env
conda env create -f envs/GroundedSam_env.yml
conda env create -f envs/control_env.yml
conda env create -f envs/instantmesh_env.yml
conda env create -f envs/pa3t2_env.yml
```

- ollama를 프롬프트 파싱에 사용하기 때문에, 아래 명령어를 통해 실행합니다.

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama run mistral:instruct
ollama serve
```

### 2. 모델 다운로드
-필요한 모델들을 다운받고, 다음 폴더에 추가합니다.

`./GroundedSAM`:[sam_vit_h_4b8939.pth](https://huggingface.co/HCMUE-Research/SAM-vit-h/blob/main/sam_vit_h_4b8939.pth), [groundingdino_swint_ogc.pth](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth)

`./ControlNet`: [Lumina3D](https://sevenstyles.com/p/lumina3d-lora-for-stable-diffusion-1-5-3135723/?srsltid=AfmBOopW3BXnMqWPS8Swlk4PGAh_8nwhJqp89twrLyXrbRK-tndOkvnp)



### 3. 실행 예시

전체 파이프라인 local 실행:
```
conda activate pa3t2
python main.py --input input_dir/your_image.jpg --prompt "A chair with a soft green leather seat" --session_name test1
```

전체 파이프라인 gradio UI로 실행:
```
conda activate pa3t2
python gradio_demo.py
```


---

## 📜 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 [`LICENSE`](./LICENSE)를 참조하세요.

---

## 🤝 모델 및 코드 기여

이 프로젝트는 다음 오픈소스 모델에 기반합니다:

- [Grounded-SAM (IDEA-Research)](https://github.com/IDEA-Research/Grounded-Segment-Anything)
- [ControlNet (lllyasviel)](https://github.com/lllyasviel/ControlNet)
- [Instant Meshes](https://github.com/TencentARC/InstantMesh)
- [Ollama (by ollama.ai)](https://ollama.com)
