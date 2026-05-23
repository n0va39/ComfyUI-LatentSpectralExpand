# ComfyUI-LatentSpectralExpand

이 노드는 **SPEED(Spectral Progressive Diffusion)** 논문([arXiv:2405.18736](https://arxiv.org/abs/2405.18736))의 아이디어 중 "Latent Spectral Expansion" 및 고주파수 노이즈 주입(High-frequency noise injection) 과정을 ComfyUI에서 실험적으로 적용해볼 수 있도록 독립적인 커스텀 노드로 구현한 것입니다.

> **주의:** 이 노드는 원본 논문의 공식 구현이 아니며, `ComfyUI-SPEED` 비공식 구현을 참고하여 
> "해상도 확장 시 주파수 도메인(DCT)에서 고주파 영역에 점진적 노이즈를 주입하는 기능"만을 별도로 구현한 실험용 모듈입니다.

## 주요 원리

기존에 단순히 픽셀이나 latent를 bicubic/nearest 방식으로 업스케일링 하는 대신, 다음의 절차를 따릅니다.
1. 입력된 저해상도 latent tensor를 **2D DCT(Discrete Cosine Transform)** 변환하여 주파수 성분으로 분해합니다.
2. 목표 해상도에 맞춰 더 큰 주파수 캔버스를 만들고, 좌상단의 저주파 영역에 기존 DCT 계수를 그대로 복사합니다.
3. 새롭게 확장된 고주파 영역에는 현재 디노이징 단계의 `sigma` 수준에 비례하는 가우시안 노이즈를 채워 넣습니다.
4. **역변환(IDCT)**을 통해 다시 공간(spatial) 영역의 latent로 되돌립니다.

이러한 방식을 통해, 업스케일 시 발생하는 블러링이나 정보 손실을 줄이고, 새 디테일 생성에 필요한 고주파 노이즈를 올바른 분포로 주입할 수 있습니다.

## 노드 입력 파라미터 (Inputs)

- `latent`: 확장할 입력 LATENT 입니다.
- `scale_factor`: (기본값: 1.25) 주파수를 확장할 **배율**입니다. 예를 들어 1.25로 설정하면 H와 W가 각각 1.25배씩 늘어납니다. 최종 해상도(H*scale_factor, W*scale_factor)는 ComfyUI의 호환성을 위해 자동으로 **가장 가까운 8의 배수로 반올림** 처리됩니다.
- `sigma`: 현재 디노이징 스텝에서 고주파 영역에 주입할 노이즈의 기준 강도입니다. 일반적인 워크플로우에서는 중간 스텝의 sigma를 직접 얻기 어려울 수 있으므로, 수동으로 적절한 값을 넣어 테스트해야 합니다.
- `noise_strength`: (기본값: 1.0) 노이즈 주입 강도의 스케일 팩터입니다. 실제 주입 강도는 `sigma * noise_strength`가 됩니다.
- `seed`: 고주파 영역 노이즈 생성의 시드값입니다. (재현성 확보용)
- `taper`: (기본값: 0) 저주파 블록과 고주파 영역 경계 부분의 코사인 감쇠(cosine drop-off) 정도를 설정합니다. 0이면 딱 잘린 형태(hard mask)를 사용하고, 4~16 정도로 주면 부드러운 전환 효과를 줍니다.
- `blend_mode`: 
  - `variance_preserving` (추천): 분산을 보존하며 taper 경계를 자연스럽게 혼합합니다.
  - `hard`: 블렌딩 없이 지정된 마스크에 따라 덮어씁니다.
  - `linear`: 단순 선형 혼합을 사용합니다.

## 시그마 분할 노드 (Sigma Split Nodes)

LSE 워크플로우에서 첫 번째 샘플러와 두 번째 샘플러 사이에 노이즈 스케줄을 자연스럽게 분할하고 정보를 연동할 수 있도록 두 가지 종류의 시그마 분할 노드를 함께 제공합니다.

### 1. Split Sigma Array (LSE)
* **공식 노드명**: `Split Sigma Array (LSE)`
* **설명**: 입력받은 전체 시그마 배열을 고정된 **정수형 스텝 수(Step Count)** 기준으로 분할합니다.
* **입력 파라미터**:
  - `sigmas`: 스케줄러에서 생성된 전체 시그마 배열입니다.
  - `split_step` (기본값: 10): 분할할 기준 스텝 인덱스입니다.

### 2. Split Sigma Array Denoise (LSE)
* **공식 노드명**: `Split Sigma Array Denoise (LSE)`
* **설명**: 입력받은 전체 시그마 배열을 **디노이즈 비율(Denoise Ratio, 0.0 ~ 1.0)** 기준으로 자동 분할합니다. ComfyUI 순정 `SplitSigmasDenoise` 노드와 동일한 수학 공식으로 동작합니다.
* **입력 파라미터**:
  - `sigmas`: 스케줄러에서 생성된 전체 시그마 배열입니다.
  - `denoise` (기본값: 0.5): 분할 기준 디노이즈 비율입니다. (예: 0.4 입력 시, 전체 스텝 중 후반부 40% 스텝을 두 번째 샘플러 영역으로 자동 할당)

---

### 📤 분할 노드 공통 출력 (Outputs)

두 분할 노드는 동일한 형태의 5가지 출력 핀을 제공하여 LSE 및 Sampler Advanced 워크플로우에 최적화되어 있습니다.

1. `high_sigmas` (`SIGMAS`)
   - **설명**: 고농도 노이즈 영역(전반부)의 시그마 배열입니다. 첫 번째 KSampler의 `sigmas` 입력에 연결합니다.
2. `low_sigmas` (`SIGMAS`)
   - **설명**: 저농도 노이즈 영역(후반부)의 시그마 배열입니다. 후속 `SetFirstSigma` 노드를 거친 후 두 번째 KSampler의 `sigmas` 입력에 연결합니다.
3. `high_steps` (`INT`)
   - **설명**: 전반부 샘플링 단계에서 실행되는 실제 스텝 수입니다.
4. `low_steps` (`INT`)
   - **설명**: 후반부 샘플링 단계에서 실행되는 실제 스텝 수입니다.
5. `transition_sigma` (`FLOAT`)
   - **설명**: 분할 지점에서의 물리적 시그마 값입니다. **`Latent Spectral Expand (LSE)` 노드의 `sigma` 입력에 바로 연결**하여 고주파 주입 노이즈의 강도를 결정하는 기준으로 사용합니다.

## 설치 방법

ComfyUI의 `custom_nodes` 디렉토리 안에 이 폴더를 위치시키면 자동으로 인식됩니다.
이 구현은 SciPy와 같은 외부 라이브러리에 의존하지 않고 오직 **PyTorch**의 기본 텐서 행렬곱을 이용하여 DCT를 처리하므로 속도가 매우 빠르고 GPU 내에서 곧바로 연산됩니다. 별도의 `requirements.txt` 설치가 필요하지 않습니다.

```bash
cd ComfyUI/custom_nodes
# 이 디렉토리(ComfyUI-LatentSpectralExpand)를 복사해 넣습니다.
```

## 사용 방법 (예시)

1. Checkpoint와 Empty Latent Image(또는 기존 이미지의 VAE Encode 결과물)를 불러옵니다.
2. 빈 Latent (예: 64x64)를 이 노드(`Latent Spectral Expand (SPEED)`)에 연결하고, `scale_factor`를 `1.25` 또는 `1.5` 등으로 설정합니다.
3. `sigma` 값을 적절히 주고 (예: 0.1~1.0 사이 테스트), `taper` 값을 8 정도로 설정합니다.
4. 확장된 Latent를 샘플러(KSampler)에 넣고 디노이징을 진행합니다.
