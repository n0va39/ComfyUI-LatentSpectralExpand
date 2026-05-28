# ComfyUI-LatentSpectralExpand

이 커스텀 노드는 **SPEED / SPD(Spectral Progressive Diffusion)** 논문([arXiv:2605.18736](https://arxiv.org/abs/2605.18736))의 아이디어 중 **latent spectral expansion**, **high-frequency noise injection**, **progressive latent resolution scheduling**을 ComfyUI에서 실험하기 위한 비공식 구현입니다.

> 이 저장소는 원본 논문의 공식 구현이 아닙니다. Anima / DiT 계열 모델에서 SPD 방식의 잠재공간 해상도 전환을 실험하기 위한 ComfyUI 노드입니다.

---

## 핵심 변화 요약

현재 버전은 Anima / DiT 사용을 전제로 다음처럼 동작합니다.

```text
최종 해상도의 latent 입력
↓
stage 0에서 DCT low-pass로 저해상도 latent 자동 생성
↓
stage 0 sampler
↓
stage 1 진입 시 spectral expand + high-frequency noise injection
↓
stage 1 sampler
↓
필요한 만큼 반복
```

중요한 기본값은 다음입니다.

```text
latent_size_multiple = 2
scheduler_mode = base_curve
transition_mode = sigma
edm_style = False
```

Anima / DiT sigma schedule은 보통 `1.0 → 0.0` 범위이므로, EDM 방식의 `sigma/(1+sigma)` 변환을 기본으로 사용하지 않습니다.

---

## 주요 원리

### 1. Stage 0: DCT low-pass initialization

사용자는 최종 해상도의 latent를 입력합니다. `LSE Stage Prepare(stage_index=0)`는 `scale_schedule[0]`에 맞춰 full latent를 DCT low-pass 방식으로 축소합니다.

```text
full latent
→ DCT
→ 좌상단 저주파 계수만 crop
→ IDCT
→ low-res latent
```

이 방식은 단순 bilinear / nearest downscale이 아니라, 주파수 공간에서 낮은 주파수 성분만 남기는 방식입니다.

### 2. Stage 1 이상: Spectral expansion

다음 stage로 넘어갈 때는 현재 latent를 DCT로 변환한 뒤, 더 큰 주파수 캔버스에 기존 계수를 복사하고 새로 열린 고주파 영역에 현재 transition sigma에 맞는 Gaussian noise를 넣습니다.

```text
low-res latent
→ DCT
→ 기존 계수는 좌상단에 보존
→ 새 고주파 영역에 sigma * noise_strength 노이즈 주입
→ IDCT
→ expanded latent
```

`variance_preserving` blend와 `taper`를 통해 경계부 혼합을 조절할 수 있습니다.

---

## 설치 방법

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/n0va39/ComfyUI-LatentSpectralExpand.git
```

SciPy 등 외부 라이브러리는 필요하지 않습니다. DCT/IDCT는 PyTorch tensor 연산으로 처리됩니다.

---

## 제공 노드

### 1. `Latent Spectral Expand (LSE)`

단일 latent 확장 노드입니다. 입력 latent를 DCT 공간에서 확장하고 새 고주파 영역에 노이즈를 주입합니다.

입력:

- `latent`: 확장할 LATENT입니다.
- `scale_factor`: 현재 latent H/W 기준 확장 배율입니다.
- `sigma`: 고주파 노이즈 주입 기준 sigma입니다.
- `noise_strength`: `sigma * noise_strength`로 주입 강도를 보정합니다.
- `seed`: 고주파 노이즈 seed입니다.
- `taper`: 저주파 영역과 고주파 영역 사이의 cosine taper 폭입니다.
- `blend_mode`: `variance_preserving`, `linear`, `hard` 중 선택합니다.
- `edm_style`: Anima / DiT에서는 기본적으로 `False`를 권장합니다.

출력:

- `latent`: 확장된 LATENT입니다.
- `sigma_aligned`: 확장 후 latent가 대응해야 하는 aligned sigma입니다.

---

## LSE_CONTEXT 기반 단계형 워크플로우

권장 워크플로우는 다음입니다.

```text
BasicScheduler
↓
LSE Segment Sigma Planner
↓
LSE Stage Prepare(stage_index=0)
↓
SamplerCustomAdvanced
↓
LSE Stage Prepare(stage_index=1)
↓
SamplerCustomAdvanced
↓
LSE Stage Prepare(stage_index=2)
↓
SamplerCustomAdvanced
↓
VAE Decode
```

`LSE Stage Prepare`는 현재 stage에 필요한 두 가지를 함께 출력합니다.

```text
processed_latent
stage_sigmas
```

따라서 stage별 sampler에는 다음처럼 연결합니다.

```text
processed_latent → SamplerCustomAdvanced latent
stage_sigmas     → SamplerCustomAdvanced sigmas
```

---

## 2. `LSE Segment Sigma Planner`

전체 SPD stage 계획을 만들고 `LSE_CONTEXT`를 출력하는 노드입니다.

### 주요 입력

- `base_sigmas`: 기준 SIGMAS입니다. 일반적으로 `BasicScheduler`의 SIGMAS를 연결합니다.
- `scale_schedule`: stage별 latent scale 목록입니다.
  - 예: `0.5,0.75,1.0`
- `transition_list`: stage 전환 지점입니다. 길이는 `len(scale_schedule) - 1`이어야 합니다.
  - 예: `0.7,0.4`
- `transition_mode`:
  - `sigma`: transition 값을 sigma로 해석합니다. Anima / DiT 기본 추천입니다.
  - `t`: transition 값을 t 값으로 직접 해석합니다.
- `scheduler_mode`:
  - `base_curve`: `base_sigmas`의 원래 곡률을 보존하며 segment별 sigma curve를 재매핑합니다. 기본 추천입니다.
  - `t_uniform`: segment 안을 t 기준 선형으로 생성합니다. 실험용입니다.
- `step_policy`:
  - `fixed_total_steps`: base SIGMAS의 총 step 수를 유지합니다.
  - `preserve_dt`: base curve상의 간격을 보존합니다. 실제 총 step 수가 증가할 수 있습니다.
- `noise_strength`, `taper`, `blend_mode`, `edm_style`: `LSE Stage Prepare`가 latent 확장 시 사용할 설정입니다.
- `seed_mode`:
  - `fixed`: 모든 stage에서 같은 seed를 사용합니다.
  - `per_stage_offset`: `seed + stage_index`를 사용합니다.
  - `random`: 실행 시 랜덤 seed를 사용합니다.

### 출력

- `lse_context`: 전체 stage 계획과 확장 설정을 담은 context입니다.
- `actual_total_steps`: 실제 배정된 총 step 수입니다.
- `segment_count`: 생성된 segment 개수입니다.

---

## 3. `LSE Stage Prepare`

`LSE_CONTEXT`, 현재 latent, `stage_index`를 받아 해당 stage에서 사용할 latent와 sigma curve를 준비합니다.

### stage 0 동작

```text
입력 latent = 최종 해상도의 full latent
↓
scale_schedule[0]에 맞춰 DCT low-pass 축소
↓
stage 0 sampler에 들어갈 low-res latent 출력
```

즉 이제는 사용자가 처음부터 low-res latent를 만들 필요가 없습니다.

### stage 1 이상 동작

```text
이전 sampler 출력 latent
↓
이전 scale → 현재 scale로 spectral expand
↓
현재 stage용 sigmas 출력
```

### 출력

- `processed_latent`: 해당 stage에 사용할 LATENT입니다.
- `stage_sigmas`: 해당 stage에 사용할 SIGMAS입니다.
- `stage_steps`: 해당 stage의 step 수입니다.
- `current_scale`: 현재 stage scale입니다.
- `transition_sigma_used`: 이 stage 진입 시 expansion에 사용한 transition sigma입니다. stage 0에서는 `0.0`입니다.
- `is_last_stage`: 마지막 stage 여부입니다.

---

## Scheduler Mode 설명

### `base_curve` 권장

`base_curve`는 `base_sigmas`의 곡선 형태를 보존합니다.

예를 들어 Anima sigma curve가 다음처럼 후반으로 갈수록 빠르게 꺾이는 형태라면:

```text
1.0000, 0.9916, 0.9828, ..., 0.0793, 0.0000
```

`base_curve`는 각 segment 내부에서도 이 원래 곡률을 최대한 유지합니다. SPD handoff로 생기는 sigma jump만 반영하고, segment 내부를 단순 직선으로 만들지 않습니다.

### `t_uniform` 실험용

`t_uniform`은 segment의 start/end 사이를 t 기준 선형으로 나눕니다. 이론 검증에는 단순하지만, Anima / DiT 실사용에서는 원래 scheduler 곡률을 깨뜨릴 수 있으므로 기본값으로 권장하지 않습니다.

---

## Step Policy 설명

### `fixed_total_steps`

base SIGMAS의 총 step 수를 유지합니다.

```text
base_sigmas가 40 step이면 모든 segment step 합도 40
```

속도 비교와 일반 사용에 적합합니다.

### `preserve_dt`

`scheduler_mode = base_curve`에서는 base curve상의 간격을 유지합니다. SPD expansion으로 전체 경로가 늘어나면 실제 총 step 수가 증가할 수 있습니다.

품질 안정성 확인용, 논문식 trajectory에 더 가까운 실험용입니다.

---

## 권장 초기 설정

Anima / DiT 3-stage 실험 기준 추천값입니다.

```text
scale_schedule = 0.5,0.75,1.0
transition_mode = sigma
transition_list = 0.7,0.4
scheduler_mode = base_curve
step_policy = fixed_total_steps
noise_strength = 1.0
taper = 8
blend_mode = variance_preserving
edm_style = false
seed_mode = per_stage_offset
```

2-stage 실험은 다음처럼 시작하면 됩니다.

```text
scale_schedule = 0.5,1.0
transition_mode = sigma
transition_list = 0.7
scheduler_mode = base_curve
step_policy = fixed_total_steps
edm_style = false
```

---

## 편의용 Sigma Segment 노드

### 4. `LSE Segment Sigmas 2`

2-stage 고정 편의 노드입니다.

```text
scale_0 → scale_1
transition 1개
segment 2개
```

주요 출력:

- `seg0_sigmas`
- `seg1_sigmas`
- `transition_sigma_0`
- `next_scale_factor_0`
- `seg0_steps`
- `seg1_steps`
- `actual_total_steps`

이 노드는 latent를 자동 확장하지 않습니다. `Latent Spectral Expand (LSE)`와 함께 쓰는 편의용 sigma 생성 노드입니다.

### 5. `LSE Segment Sigmas 3`

3-stage 고정 편의 노드입니다.

```text
scale_0 → scale_1 → scale_2
transition 2개
segment 3개
```

주요 출력:

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

### 7. `Split Sigma Array Denoise (LSE)`

입력 SIGMAS를 denoise 비율 기준으로 둘로 나눕니다. ComfyUI 순정 `SplitSigmasDenoise`와 같은 방식입니다.

---

## 주의 사항

- Anima / DiT 전용으로 생각하고 `latent_size_multiple = 2`로 처리합니다.
- `transition_list`는 반드시 감소해야 합니다.
  - 예: `0.7,0.4` 가능
  - `0.4,0.7` 불가능
- `len(transition_list) = len(scale_schedule) - 1`이어야 합니다.
- Stage Prepare 방식에서는 `Latent Spectral Expand (LSE)` 노드를 별도로 연결하지 않습니다.
- `edm_style = false`가 Anima / DiT 기본 추천입니다.
- `noise_strength = 1.0`이 논문식 기본 가정에 가장 가깝습니다.
- `preserve_dt`는 품질 안정성에는 유리할 수 있지만 실제 step 수가 증가할 수 있습니다.

---

## Codex handoff notes

다음 작업은 Codex에서 이어서 진행하기 좋습니다.

### 현재 구현 상태

구현된 주요 기능:

- `LSE_CONTEXT` 기반 stage planning
- stage 0 DCT low-pass initialization
- stage 1 이상 DCT spectral expansion
- Anima / DiT용 latent H/W 2배수 snap
- `scheduler_mode = base_curve`
- `scheduler_mode = t_uniform`
- `step_policy = fixed_total_steps`
- `step_policy = preserve_dt`
- `transition_mode = sigma` / `t`
- `edm_style = false` 기본값

핵심 파일:

```text
latent_spectral_expand.py
__init__.py
README.md
```

### base_curve 구현 의도

`t_uniform`은 segment 내부 sigma를 선형으로 만들어 원래 Anima sigma curve를 과하게 깨뜨렸습니다. 그래서 `base_curve`를 추가했습니다.

`base_curve`는 다음 원칙을 따릅니다.

```text
1. base_sigmas를 원래 scheduler template로 본다.
2. sigma 또는 t transition을 base curve상의 위치 u로 변환한다.
3. 각 segment는 start_u → end_u 구간을 다시 샘플링한다.
4. segment 내부 곡률은 base_sigmas의 원래 형태를 따른다.
5. SPD expansion으로 생기는 aligned start만 반영한다.
```

### Codex에게 맡길 다음 검증 작업

1. **ComfyUI import 검증**
   - ComfyUI 실행 시 모든 노드가 정상 등록되는지 확인합니다.
   - 특히 `LSE_CONTEXT` 타입과 `BOOLEAN` 출력이 문제 없는지 확인합니다.

2. **shape 검증**
   - full latent 예: `128×128`
   - `scale_schedule = 0.5,0.75,1.0`
   - stage 0 출력 latent가 `64×64`가 되는지 확인합니다.
   - stage 1 출력 latent가 `96×96`가 되는지 확인합니다.
   - stage 2 출력 latent가 `128×128`가 되는지 확인합니다.

3. **sigma curve 검증**
   - `base_sigmas`가 `1.0 → 0.0` 곡선일 때:
     - stage 0은 `1.0 → transition_0`
     - stage 1은 `align(transition_0, scale_1/scale_0) → transition_1`
     - stage 2는 `align(transition_1, scale_2/scale_1) → 0.0`
   - `scheduler_mode = base_curve`에서 segment 내부 곡률이 원래 base curve와 비슷한지 plot으로 확인합니다.

4. **edm_style 검증**
   - Anima / DiT에서는 `edm_style=False`가 기본입니다.
   - `edm_style=True`는 EDM sigma가 1보다 큰 모델용 실험 옵션으로만 남깁니다.

5. **workflow 검증**
   - `LSE Segment Sigma Planner`
   - `LSE Stage Prepare(stage_index=0)`
   - `SamplerCustomAdvanced`
   - `LSE Stage Prepare(stage_index=1)`
   - `SamplerCustomAdvanced`
   - `LSE Stage Prepare(stage_index=2)`
   - `SamplerCustomAdvanced`

6. **후속 개선 후보**
   - `LSE_CONTEXT` 내용을 표시하는 debug/info 노드 추가
   - stage별 latent shape와 sigma start/end를 출력하는 debug 노드 추가
   - 통합 sampler 구현
   - Euler 전용 통합 sampler부터 구현
   - 이후 ER-SDE / DPM++ 계열은 실험 옵션으로 분리
   - `base_curve` 보간 방식을 sigma-linear가 아니라 logit/timestep 기반으로 개선할지 비교
   - `taper`와 `variance_preserving` 기본값 검증

### 현재 기본 추천값

```text
scale_schedule = 0.5,0.75,1.0
transition_mode = sigma
transition_list = 0.7,0.4
scheduler_mode = base_curve
step_policy = fixed_total_steps
edm_style = false
noise_strength = 1.0
taper = 8
blend_mode = variance_preserving
seed_mode = per_stage_offset
```

### 알려진 설계 판단

- 이 구현은 Anima / DiT 중심입니다.
- latent 크기는 2의 배수로 snap합니다.
- `base_curve`가 기본 scheduler mode입니다.
- `t_uniform`은 실험용으로만 유지합니다.
- Stage Prepare 방식에서는 full-size latent를 입력하고, stage 0에서 자동으로 DCT low-pass 축소합니다.
- 통합 sampler는 아직 구현하지 않았습니다.
