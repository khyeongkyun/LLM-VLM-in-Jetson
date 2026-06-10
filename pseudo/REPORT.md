# Mistral-Small-3.1-24B-Instruct 양자화 비교 보고서
**GPTQ vs FOEM, 4-bit & 3-bit (Weight-only, group_size=128)**

작성일: 2026-05-28 · 작업 디렉토리: `/workspace/pseudo`

---

## 1. 목적

로컬 HF 캐시에 받아둔 `mistralai/Mistral-Small-3.1-24B-Instruct-2503` (멀티모달, ~48 GB / bf16) 의 **언어 백본**을 다음 네 가지 설정으로 양자화하고 WikiText-2 perplexity 로 품질을 비교한다.

| 방법 | 설명 |
|---|---|
| GPTQ (일반) | `QuantizeConfig(bits=B, group_size=128)` |
| FOEM | `QuantizeConfig(bits=B, group_size=128, foem=FOEMConfig(alpha=0, beta=0.2))` |

(B ∈ {4, 3}; 총 4 모델)

비전 타워와 multimodal projector 는 fp16 으로 보존하여 멀티모달 기능을 유지.

---

## 2. 방법론

### 2.1 FOEM (First-Order Error Matters, AAAI 2026)

`/workspace/pseudo/FOEM` 의 README 가 명시하듯, FOEM 의 권장 사용 경로는 **GPTQModel 통합본** 이다. 본 작업 시점에 FOEM 은 upstream `gptqmodel` (PyPI) 에 `FOEMConfig` 로 정식 병합되어 있어, 별도 포크 없이 `pip install gptqmodel` 만으로 사용 가능했다.

FOEM 은 GPTQ 의 가중치 갱신식에 두 가지 보정항을 추가한다:

```
ΔW = -e_i ⊗ H⁻¹[i,:]            (기본 GPTQ 항)
     + w ⊗ P[i,:]                (첫째 차수 보정, P = α·((dXXᵀ·H⁻ᵀ).triu)·H⁻¹)
     - (W - W_fp) · (H⁻²) · β    (직접 오차 피드백)
```

본 실험은 README 의 기본 권장 설정인 **α=0.0, β=0.2 (FOEM w/o GPTAQ)** 를 사용. α=0 이므로 첫째 차수 보정은 끄고 β 항만 활성화되는 형태이다.

### 2.2 일반 GPTQ

위 식에서 두 보정항을 모두 끈 형태 (`foem` 인자 미지정). 동일 코드 경로, 동일 캘리브레이션 데이터를 사용하므로 **방법 외 변수가 완벽히 통제**된 비교이다.

### 2.3 양자화 대상 범위

대상 모델은 `Mistral3ForConditionalGeneration` (`model_type=mistral3`) 으로, 비전 인코더 + 텍스트 LLM 구조이다. GPTQModel 7.0.0 은 `mistral3` 를 레지스트리에 등록 (`Mistral3GPTQ`) 하며, **언어 백본 (`model.language_model.layers.*`) 의 4 종 어텐션 + 3 종 MLP linear 만 양자화**하고 비전 타워와 projector 는 건드리지 않는다.

---

## 3. 실험 설정

### 3.1 하드웨어 및 소프트웨어 환경

| 항목 | 값 |
|---|---|
| GPU | NVIDIA RTX A6000 (49 GB) × 2 + RTX 5000 Ada (32 GB) × 2 — **A6000 1장 단독 사용** |
| CPU RAM | 503 GB |
| OS / driver | Linux, CUDA 12.4, 드라이버 550.90.07 |
| Python / conda env | 3.11, `/opt/conda/envs/newmjx` |
| 주요 패키지 | `gptqmodel==7.0.0`, `transformers==5.9.0`, `accelerate==1.13.0`, `torch==2.9.1+cu128`, `datasets==4.8.5`, `sentencepiece==0.2.1`, `tiktoken==0.13.0` |

### 3.2 캘리브레이션

| 항목 | 값 |
|---|---|
| 데이터셋 | `allenai/c4` (en, `c4-train.00001-of-01024.json.gz`) |
| 샘플 수 | 256 |
| 토큰 수 (실측) | non-padded 115,950 / 총 132,840 (256 샘플 배치) |

### 3.3 양자화 하이퍼파라미터

| 항목 | GPTQ | FOEM |
|---|---|---|
| bits | 3, 4 | 3, 4 |
| group_size | 128 | 128 |
| `desc_act` | False | False |
| `damp_percent` | 0.05 | 0.05 |
| `attn_implementation` | sdpa | sdpa |
| `offload_to_disk` | False | False |
| α (alpha) | — | 0.0 |
| β (beta) | — | 0.2 |

### 3.4 평가

직접 구현한 표준 sliding-window perplexity (`/workspace/pseudo/eval_ppl.py`):

- 데이터: WikiText-2 raw v1 (test split), `"\n\n".join(text)` 으로 연결 후 토큰화.
- 윈도우: `seqlen=2048`, 비중첩 슬라이딩 (= `n = len(tokens) // 2048` 개 윈도우).
- 각 윈도우의 cross-entropy 를 합산 후 `exp(mean NLL)`.
- bf16 forward 에서 발생한 NaN/Inf 윈도우는 제외 (양자화 모델에서는 발생하지 않음).

`lm-eval` 의 모델 래퍼는 mistral3 멀티모달 양자화 모델과의 호환성이 불확실하여 사용하지 않고 직접 구현. 측정 지표 (WikiText-2 PPL) 자체는 FOEM 논문 / README 표와 동일.

---

## 4. 결과

### 4.1 PPL 비교 (WikiText-2 test, seqlen=2048)

| 방법 | bits | **PPL** | 디스크 |
|---|---:|---:|---:|
| GPTQ (일반) | 4 | **5.6093** | 15 GB |
| FOEM (α=0, β=0.2) | 4 | **5.6166** | 15 GB |
| GPTQ (일반) | 3 | **6.6027** | 12 GB |
| FOEM (α=0, β=0.2) | 3 | **6.6526** | 12 GB |
| fp16 (참고) | 16 | — (※) | 45 GB |

(※) fp16 베이스라인은 Mistral-Small-3.1 의 bf16 long-context forward 에서 활성값 NaN/Inf 가 발생해 정상 값 측정 실패. 양자화 모델은 Marlin 커널이 dequant 를 fp32 로 처리하므로 영향을 받지 않음. (필요시 fp32 또는 별도 워크어라운드로 재측정 가능.)

### 4.2 양자화 품질 점검 — RTN 폴백률

GPTQ/FOEM 은 Hessian 이 non-PD 일 때 RTN (round-to-nearest) 으로 폴백한다. 폴백된 모듈은 GPTQ/FOEM 효과가 사라지므로 비교가 의미를 잃는다.

| 모델 | RTN 폴백 모듈 수 / 전체 | 평균 damp |
|---|---:|---:|
| GPTQ 4-bit | **0** / 280 | 0.05 |
| FOEM 4-bit | **0** / 280 | 0.05 |
| GPTQ 3-bit | **0** / 280 | 0.05 |
| FOEM 3-bit | **0** / 280 | 0.05 |

(40 레이어 × 7 모듈 = 280)

### 4.3 양자화 손실 분포 (레이어 0, GPTQ 4-bit, 참고용)

| 모듈 | shape | loss (∥WX − W_q X∥²) |
|---|---|---:|
| self_attn.q_proj | 5120 → 4096 | 1.39 × 10⁻⁶ |
| self_attn.k_proj | 5120 → 1024 | 6.51 × 10⁻⁷ |
| self_attn.v_proj | 5120 → 1024 | 2.36 × 10⁻⁸ |
| self_attn.o_proj | 4096 → 5120 | 3.00 × 10⁻¹⁰ |
| mlp.gate_proj | 5120 → 32768 | 2.27 × 10⁻⁶ |
| mlp.up_proj | 5120 → 32768 | 2.09 × 10⁻⁶ |
| mlp.down_proj | 32768 → 5120 | 1.10 × 10⁻⁹ |

전형적인 GPTQ 손실 스케일 (10⁻⁶ ~ 10⁻¹⁰) — 정상.

---

## 5. 분석

### 5.1 4-bit 결과

GPTQ 5.6093, FOEM 5.6166 — **차이 +0.007 (0.13 %)**. 노이즈 수준의 격차로, 사실상 동등. FOEM 원논문 (Qwen3-8B 4-bit) 의 GPTQ 12.55 → FOEM 12.51 격차 (~0.04) 와 같은 자릿수의 변동이지만 본 실험에서는 방향이 반대로 나옴.

### 5.2 3-bit 결과

GPTQ 6.6027, FOEM 6.6526 — **차이 +0.050 (0.76 %)**. 여전히 작지만 4-bit 보다 변동성이 커짐. 더 공격적인 양자화에서 FOEM 의 β=0.2 보정이 본 모델·캘리브레이션 조합에 최적이 아닐 가능성.

### 5.3 종합

- **두 방법 모두 작동이 검증됨** (RTN 폴백 0건, 손실 정상 스케일, 생성 텍스트 정상).
- 본 모델·설정에서는 FOEM 의 우위가 관찰되지 않았고 오히려 미세하게 열세. 다만 격차는 실용적으로 무의미한 수준.
- FOEM 의 권장 hyperparameter (α=0, β=0.2) 는 Qwen 계열에서 검증된 값으로, Mistral-Small-3.1 에는 추가 튜닝 여지가 있음.

### 5.4 후속 실험 제안 (필요 시)

| 방향 | 변경점 | 기대 효과 |
|---|---|---|
| FOEM hyperparameter 탐색 | β ∈ {0.05, 0.1, 0.3}, α ∈ {0.0, 0.25} | 모델별 최적 β 발견 |
| GPTAQ 변형 | α=0.25, β=0.2 (FOEM with GPTAQ) | 첫째 차수 보정 효과 검증 |
| 캘리브레이션 확대 | 512 ~ 1024 샘플 | 분산 감소, 격차의 통계적 유의성 검증 |
| 더 공격적 bit | 2-bit | 보정항 효과가 두드러질 가능성 |
| 다른 평가 데이터 | C4 perplexity, lm-eval (mmlu 등) | task-level 성능 비교 |

---

## 6. 구현 — 산출물 및 환경 메모

### 6.1 파일

| 경로 | 역할 |
|---|---|
| `/workspace/pseudo/quantize_mistral.py` | 양자화 스크립트 (`--method gptq|foem --bits 3|4 --attn sdpa`) |
| `/workspace/pseudo/eval_ppl.py` | WikiText-2 PPL 직접 측정 |
| `/workspace/pseudo/quantized/MistralSmall3.1-24B-{gptq,foem}-{4,3}bit/` | 양자화 모델 4종 (각 12 ~ 15 GB) |
| `/workspace/pseudo/logs_{gptq,foem}-{4,3}bit.log` | 양자화 실행 로그 |
| `/workspace/pseudo/eval_{gptq,foem}-{4,3}bit.log` | 평가 로그 |

### 6.2 환경에 가해진 변경 (사용자 인지 필요)

- `gptqmodel==7.0.0` 설치 시 의존성으로 `transformers` 가 **4.57.3 → 5.9.0** 으로 자동 메이저 업그레이드. `tensorrt-edgellm==0.5.0` (transformers==4.57.3 핀) 이 깨질 수 있음.
- 같은 작업 환경에서 다른 프로젝트가 영향받지 않으려면 **분리된 venv/conda env** 사용 권장.
- 캐시에 누락돼 있던 토크나이저 / 프로세서 파일 (`tokenizer.json`, `tekken.json`, `preprocessor_config.json` 등) 을 `snapshot_download(ignore_patterns=["*.safetensors"])` 로 보충함.
- 추가 설치: `sentencepiece`, `tiktoken`, `lm-eval`, `datasets`.

### 6.3 디버깅 과정에서 발견된 비공식 제약 (재현/이식 시 주의)

| 문제 | 증상 | 해결 |
|---|---|---|
| 디스크 오프로드 | `Cannot copy out of meta tensor` | `QuantizeConfig(offload_to_disk=False)` |
| `attn=eager` + 긴 시퀀스 | CUDA OOM (29 GB 단일 할당) | `attn_implementation="sdpa"` |
| **멀티 GPU 데이터병렬 forward** | downstream subset 활성값이 NaN → ~50 % 모듈이 RTN 으로 폴백 | **단일 GPU** 강제 (`CUDA_VISIBLE_DEVICES=<one>`) |
| `tie_word_embeddings` | 저장된 config 에 `True` 로 잘못 설정 → 로드 시 lm_head 가 embed_tokens 로 묶여 생성이 garbage | 저장 후 `config.json` 의 `tie_word_embeddings: false` 패치 |
| `wikitext` 데이터셋 ID | 새 datasets 가 `namespace/name` 강제 | `Salesforce/wikitext` 사용 |

가장 영향이 큰 항목은 **단일 GPU 제약** 이다. 멀티 GPU 사용 시 RTN 폴백률이 ~50 % 까지 치솟아 GPTQ/FOEM 비교가 무의미해진다 (실제 본 작업의 초기 시도에서 관찰).

---

## 7. 결론

- 이 폴더의 FOEM 방법론과 일반 GPTQ 방법론을 동일 코드 경로 (`gptqmodel.QuantizeConfig` 의 `foem=` 인자 유무) 로 일관되게 적용, Mistral-Small-3.1-24B 의 언어 백본을 4-bit / 3-bit 로 양자화하고 WikiText-2 PPL 로 비교했다.
- 4개 모델 전부 깨끗하게 양자화 완료 (RTN 폴백 0건, 정상 PPL 5.6 ~ 6.7).
- 본 모델·설정에서 두 방법의 PPL 격차는 4-bit 0.007 / 3-bit 0.050 으로 **사실상 동등**, FOEM 의 명확한 우위는 관찰되지 않음.
- 결정적 발견: **단일 GPU + sdpa + offload_to_disk=False + tie_word_embeddings 패치** 가 본 환경 (gptqmodel 7.0.0 + transformers 5.9.0 + torch 2.9.1) 에서 깨끗한 양자화·평가에 필수.

---
