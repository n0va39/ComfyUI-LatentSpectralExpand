# ComfyUI-LatentSpectralExpand

이 커스텀 노드는 **SPEED / SPD(Spectral Progressive Diffusion)** 논문([arXiv:2605.18736](https://arxiv.org/abs/2605.18736))의 아이디어 중 **latent spectral expansion**과 **high-frequency noise injection**을 ComfyUI에서 실험할 수 있도록 만든 비공식 구현입니다.

> **주의:** 이 저장소는 원본 논문의 공식 구현이 아닙니다. 논문 아이디어와 `ComfyUI-SPEED` 비공식 구현을 참고하여, ComfyUI에서 단계별 latent 해상도 확장과 sigma schedule 실험을 할 수 있도록 구성한 실험용 노드입니다.

---

## 주요 원리

단순 bicubic/nearest latent upscale 대신 다음 과정을 사용합니다.

1. 입력 latent tensor의 H/W 공간축에 대해 channel별 **2D DCT(Discrete Cosine Transform)**를 수행합니다.
2. 목표 latent 해상도에 맞춰 더 큰 DCT 주파수 캔버스를 만듭니다.
3. 기존 DCT 계수는 좌상단 저주파 영역에 복사합니다.
4. 새로 열린 고주파 영역에는 현재 denoising 단계의 `sigma`에 비례하는 Gaussian noise를 주입합니다.
5. IDCT로 다시 spatial latent로 되돌립니다.

즉, 기존 저주파 구조는 보존하고, 새 해상도에서만 표현 가능한 고주파 대역을 현재 noise level에 맞는 노이즈로 열어 주는 방식입니다.

---

## 설치 방법

ComfyUI의 `custom_nodes` 디렉토리 안에 이 폴더를 넣으면 됩니다.

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/n0va39/ComfyUI-LatentSpectralExpand.git
```

SciPy 등 외부 라이브러리는 필요하지 않습니다. DCT/IDCT는 PyTorch tensor 연산으로 처리됩니다.

---

## 제공 노드 목록

### 1. `Latent Spectral Expand (LSE)`

기본 latent 확장 노드입니다. 입력 latent를 DCT 공간에서 확장하고, 새 고주파 영역에 `sigma * noise_strength` 수준의 노이즈를 주입합니다.

#### 입력

- `latent`: 확장할 LATENT입니다.
- `scale_factor`: latent H/W를 확장할 배율입니다. 예: `1.25`, `1.5`, `2.0`.
- `sigma`: 고주파 노이즈 주입 기준 sigma입니다.
- `noise_strength`: 노이즈 주입 강도 보정값입니다. 실제 주입 강도는 `sigma * noise_strength`입니다.
- `seed`: 고주파 노이즈 생성 seed입니다.
- `taper`: 저주파 보존 영역과 새 고주파 영역 사이의 cosine taper 폭입니다. `0`이면 hard mask입니다.
- `blend_mode`:
  - `variance_preserving`: taper 경계에서 분산 보존형 혼합을 사용합니다.
  - `linear`: 단순 선형 혼합입니다.
  - `hard`: mask 기준 혼합입니다.
- `edm_style`: `sigma -> t = sigma / (1 + sigma)` 변환을 사용합니다.

#### 출력

- `latent`: 확장된 LATENT입니다.
- `sigma_aligned`: 확장 후 latent가 대응해야 하는 aligned sigma입니다.

---

## LSE Context 기반 단계형 워크플로우

새 버전에서는 단순 sigma split 외에 **LSE_CONTEXT** 기반 단계형 워크플로우를 제공합니다.

핵심 구조는 다음과 같습니다.

```text
LSE Segment Sigma Planner
→ LSE Stage Prepare(stage_index=0)
→ SamplerCustomAdvanced
→ LSE Stage Prepare(stage_index=1)
→ SamplerCustomAdvanced
→ LSE Stage Prepare(stage_index=2)
→ SamplerCustomAdvanced
```

`LSE Stage Prepare`는 현재 stage에 필요한 sigma curve를 꺼내고, `stage_index > 0`이면 이전 stage 결과 latent를 자동으로 spectral expand한 뒤 출력합니다.

---

### 2. `LSE Segment Sigma Planner`

전체 progressive sampling 계획을 만들고 `LSE_CONTEXT`를 출력하는 노드입니다.

이 노드는 기존 sigma 배열을 단순히 자르지 않고, 논문식 `t` 개념에 가까운 `t_uniform` 방식으로 각 segment의 sigma curve를 새로 생성합니다.

#### 입력

- `base_sigmas`: 기준 SIGMAS입니다. 일반 BasicScheduler 등의 출력 SIGMAS를 연결합니다.
- `scale_schedule`: stage별 latent scale 목록입니다. 콤마 문자열로 입력합니다.
  - 예: `0.5,0.75,1.0`
- `transition_list`: stage 전환 지점 목록입니다. 길이는 `scale_schedule 개수 - 1`이어야 합니다.
  - 예: `0.55,0.22`
- `transition_mode`:
  - `t`: `transition_list`를 t 값으로 해석합니다.
  - `sigma`: `transition_list`를 sigma 값으로 해석한 뒤 내부에서 t로 변환합니다.
- `step_policy`:
  - `fixed_total_steps`: base sigma의 총 step 수를 유지합니다.
  - `preserve_dt`: 원래 t 간격을 유지합니다. 실제 total step이 증가할 수 있습니다.
- `scheduler_mode`: 현재는 `t_uniform`만 지원합니다.
- `noise_strength`, `taper`, `blend_mode`, `edm_style`: 이후 `LSE Stage Prepare`가 latent 확장 시 사용할 설정입니다.
- `seed_mode`:
  - `fixed`: 모든 stage 확장에서 같은 seed를 사용합니다.
  - `per_stage_offset`: `seed + stage_index`를 사용합니다. 기본 추천값입니다.
  - `random`: 실행 시 랜덤 seed를 사용합니다.
- `seed`: 기본 seed입니다.

#### 출력

- `lse_context`: 전체 segment 계획과 latent 확장 설정을 담은 `LSE_CONTEXT`입니다.
- `actual_total_steps`: 실제 배정된 전체 step 수입니다.
  - `fixed_total_steps`에서는 보통 base step과 같습니다.
  - `preserve_dt`에서는 더 커질 수 있습니다.
- `segment_count`: 생성된 segment 개수입니다.

---

### 3. `LSE Stage Prepare`

`LSE_CONTEXT`와 현재 latent, stage index를 받아 해당 stage에서 사용할 sampler 입력을 준비합니다.

#### 입력

- `lse_context`: `LSE Segment Sigma Planner`의 출력입니다.
- `latent`: 현재 단계에 들어갈 LATENT입니다.
  - `stage_index = 0`이면 원본 latent를 넣습니다.
  - `stage_index > 0`이면 이전 SamplerCustomAdvanced의 출력 latent를 넣습니다.
- `stage_index`: 현재 stage 번호입니다. `0`부터 시작합니다.

#### 동작

- `stage_index = 0`:
  - latent는 그대로 통과시킵니다.
  - stage 0용 `stage_sigmas`를 출력합니다.
- `stage_index > 0`:
  - 이전 scale에서 현재 scale로 latent spectral expand를 자동 수행합니다.
  - 이전 transition sigma를 이용해 고주파 노이즈를 주입합니다.
  - 현재 stage용 `stage_sigmas`를 출력합니다.

#### 출력

- `processed_latent`: 해당 stage에 사용할 LATENT입니다.
- `stage_sigmas`: 해당 stage에 사용할 SIGMAS입니다.
- `stage_steps`: 해당 stage의 step 수입니다.
- `current_scale`: 현재 stage scale입니다.
- `transition_sigma_used`: 이 stage 진입 시 latent 확장에 사용된 transition sigma입니다. stage 0에서는 `0.0`입니다.
- `is_last_stage`: 마지막 stage 여부입니다.

---

## Stage Prepare 사용 예시

3-stage 예시는 다음과 같습니다.

```text
BasicScheduler
↓
LSE Segment Sigma Planner
  scale_schedule = 0.5,0.75,1.0
  transition_list = 0.55,0.22
  transition_mode = t
  step_policy = fixed_total_steps

Empty Latent 또는 시작 latent
↓
LSE Stage Prepare(stage_index=0)
  processed_latent → SamplerCustomAdvanced latent
  stage_sigmas → SamplerCustomAdvanced sigmas
↓
SamplerCustomAdvanced #0
↓
LSE Stage Prepare(stage_index=1)
  processed_latent → SamplerCustomAdvanced latent
  stage_sigmas → SamplerCustomAdvanced sigmas
↓
SamplerCustomAdvanced #1
↓
LSE Stage Prepare(stage_index=2)
  processed_latent → SamplerCustomAdvanced latent
  stage_sigmas → SamplerCustomAdvanced sigmas
↓
SamplerCustomAdvanced #2
↓
VAE Decode
```

---

## Step Policy 설명

### `fixed_total_steps`

base SIGMAS의 총 step 수를 유지합니다.

```text
예: base_sigmas가 30 step이면, 모든 segment step 합도 30
```

확장 후 aligned t 경로까지 포함한 전체 t 길이에 비례해서 각 segment step을 배분합니다. 속도 비교와 일반 사용에 적합합니다.

### `preserve_dt`

base SIGMAS의 평균 t 간격을 유지합니다.

```text
예: base 30 step이더라도 확장으로 t 경로가 길어지면 실제 total step은 34, 36 등으로 증가 가능
```

논문식 t trajectory에 더 가까운 품질/재현 실험에 적합합니다.

---

## 편의용 Sigma Segment 노드

### 4. `LSE Segment Sigmas 2`

2-stage 고정 편의 노드입니다.

```text
scale_0 → scale_1
transition 1개
segment 2개
```

#### 주요 출력

- `seg0_sigmas`
- `seg1_sigmas`
- `transition_sigma_0`
- `next_scale_factor_0`
- `seg0_steps`
- `seg1_steps`
- `actual_total_steps`

이 노드는 latent를 자동 확장하지 않습니다. 기존 `Latent Spectral Expand (LSE)`와 함께 쓰는 편의용 sigma 생성 노드입니다.

### 5. `LSE Segment Sigmas 3`

3-stage 고정 편의 노드입니다.

```text
scale_0 → scale_1 → scale_2
transition 2개
segment 3개
```

#### 주요 출력

- `seg0_sigmas`
- `seg1_sigmas`
- `seg2_sigmas`
- `transition_sigma_0`
- `transition_sigma_1`
- `next_scale_factor_0`
- `next_scale_factor_1`
- `seg0_steps`
- `seg1_steps`
- `seg2_steps`
- `actual_total_steps`

---

## 기존 Sigma Split 노드

기존 호환성을 위해 단순 split 노드도 유지합니다.

### 6. `Split Sigma Array (LSE)`

입력 SIGMAS를 정수형 `split_step` 기준으로 둘로 나눕니다.

출력:

- `high_sigmas`
- `low_sigmas`
- `high_steps`
- `low_steps`
- `transition_sigma`

### 7. `Split Sigma Array Denoise (LSE)`

입력 SIGMAS를 denoise 비율 기준으로 둘로 나눕니다. ComfyUI 순정 `SplitSigmasDenoise`와 같은 방식입니다.

출력:

- `high_sigmas`
- `low_sigmas`
- `high_steps`
- `low_steps`
- `transition_sigma`

---

## 권장 초기 설정

3-stage 실험 기준으로 다음을 권장합니다.

```text
scale_schedule = 0.5,0.75,1.0
transition_mode = t
transition_list = 0.55,0.22
step_policy = fixed_total_steps
scheduler_mode = t_uniform
noise_strength = 1.0
taper = 8
blend_mode = variance_preserving
edm_style = true
seed_mode = per_stage_offset
```

품질을 더 안정적으로 보고 싶으면 `step_policy = preserve_dt`를 사용합니다.

---

## 주의 사항

- 이 노드는 공식 SPEED 구현이 아닙니다.
- `transition_list`는 반드시 감소해야 합니다.
  - 예: `0.55,0.22`는 가능하지만 `0.22,0.55`는 불가능합니다.
- `len(transition_list) = len(scale_schedule) - 1`이어야 합니다.
- `LSE Stage Prepare`의 `stage_index`는 `0`부터 시작합니다.
- Stage Prepare 방식에서는 `Latent Spectral Expand (LSE)` 노드를 별도로 연결하지 않습니다.
- 편의용 `LSE Segment Sigmas 2/3`는 sigma만 생성하므로 latent 확장은 별도 노드가 필요합니다.
- `noise_strength = 1.0`이 논문식 기본 가정에 가장 가깝습니다.
- `preserve_dt`는 품질 안정성에는 유리할 수 있지만 실제 step 수가 증가할 수 있습니다.
