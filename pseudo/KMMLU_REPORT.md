# Ministral-3-3B KMMLU 평가 비교 보고서
**BF16 원본 vs GPTQ 4-bit vs FOEM 3-bit (5-shot, KMMLU 45개 과목 전체)**

작성일: 2026-07-01 · 모델: `mistralai/Ministral-3-3B-Instruct-2512-BF16`

---

## 1. 목적

WikiText-2 perplexity와 KMMLU 정확도를 세 가지 모델 변형에 걸쳐 비교한다: ① **BF16 원본** (비양자화, 기준선), ② **GPTQ 4-bit**, ③ **FOEM 3-bit** (α=0, β=0.2). 이를 통해 각 양자화 방법이 한국어 도메인 지식·추론 능력을 얼마나 보존/손실시키는지 정량화한다.

| 모델 | 양자화 방법 | 비트 | WikiText-2 PPL |
|---|---|---|---|
| `Ministral-3-3B-Instruct-2512-BF16` | 없음 (기준) | BF16 | N/A |
| `Ministral-3-3B-Instruct-2512-BF16_gptq_4bit` | GPTQ | 4-bit | 8.72 |
| `Ministral-3-3B-Instruct-2512-BF16_foem_3bit` | FOEM (α=0, β=0.2) | 3-bit | 10.87 |

---

## 2. 평가 방법

- **데이터셋**: `HAERAE-HUB/KMMLU` — 단일 "All" config 없음, 45개 과목별 config 순회 누적.
- **Few-shot**: 5-shot (각 과목 `dev` split 5문항, KMMLU 공식 벤치마크와 동일 설정).
- **채점 방식**: log-likelihood 기반 4지선다 (generate() 없음). 5-shot 프롬프트 뒤에서 " A"/" B"/" C"/" D" 토큰의 logit argmax.
- **구현**: `pseudo/eval_kmmlu.py`. BF16 원본은 `Mistral3ForConditionalGeneration.from_pretrained()` + `tie_weights()` 호출, 양자화 모델은 `GPTQModel.load()` + `.model` 추출 패턴.

---

## 3. 전체 결과

| 모델 | 정확도 (micro) | 정확도 (macro) | 소요 시간 | 문항 수 |
|---|---:|---:|---:|---:|
| **BF16 원본** | **46.88%** | 46.02% | 47.1분 | 35,030 |
| **GPTQ 4-bit** | **44.19%** | 43.42% | 62.4분 | 35,030 |
| **FOEM 3-bit** | **35.58%** | 35.03% | 64.9분 | 35,030 |
| 4지선다 무작위 기준선 | 25.00% | 25.00% | - | - |

**양자화로 인한 정확도 저하 (BF16 대비)**

| 모델 | micro 차이 | macro 차이 |
|---|---:|---:|
| GPTQ 4-bit | -2.69%p | -2.60%p |
| FOEM 3-bit | -11.30%p | -10.99%p |

---

## 4. 과목별 분석

<details><summary>45개 과목 전체 3-way 비교 표</summary>

| 과목 | BF16 | GPTQ 4-bit | FOEM 3-bit | 문항 수 |
|---|---:|---:|---:|---:|
| Accounting | 41.0% | 37.0% | 37.0% | 100 |
| Agricultural-Sciences | 37.7% | 36.1% | 28.0% | 1000 |
| Aviation-Engineering-and-Maintenance | 44.7% | 41.5% | 35.3% | 1000 |
| Biology | 38.9% | 36.9% | 28.8% | 1000 |
| Chemical-Engineering | 49.8% | 47.3% | 37.0% | 1000 |
| Chemistry | 51.8% | 48.0% | 36.5% | 600 |
| Civil-Engineering | 44.2% | 41.5% | 31.3% | 1000 |
| Computer-Science | 72.0% | 69.2% | 54.1% | 1000 |
| Construction | 36.8% | 34.4% | 32.3% | 1000 |
| Criminal-Law | 34.0% | 33.5% | 27.5% | 200 |
| Ecology | 48.7% | 45.8% | 31.4% | 1000 |
| Economics | 46.9% | 46.9% | 39.2% | 130 |
| Education | 64.0% | 58.0% | 37.0% | 100 |
| Electrical-Engineering | 35.3% | 34.3% | 26.4% | 1000 |
| Electronics-Engineering | 54.5% | 52.9% | 37.9% | 1000 |
| Energy-Management | 34.0% | 33.1% | 27.5% | 1000 |
| Environmental-Science | 35.0% | 30.4% | 25.2% | 1000 |
| Fashion | 46.4% | 44.6% | 35.1% | 1000 |
| Food-Processing | 43.6% | 41.9% | 34.4% | 1000 |
| Gas-Technology-and-Engineering | 39.9% | 36.4% | 30.3% | 1000 |
| Geomatics | 44.5% | 39.8% | 32.6% | 1000 |
| Health | 64.0% | 55.0% | 37.0% | 100 |
| Industrial-Engineer | 44.3% | 40.5% | 33.0% | 1000 |
| Information-Technology | 69.1% | 64.4% | 49.1% | 1000 |
| Interior-Architecture-and-Design | 53.2% | 50.8% | 41.2% | 1000 |
| Law | 45.7% | 42.6% | 33.3% | 1000 |
| Machine-Design-and-Manufacturing | 44.4% | 42.8% | 34.4% | 1000 |
| Management | 53.9% | 49.0% | 41.6% | 1000 |
| Maritime-Engineering | 47.0% | 44.8% | 35.2% | 600 |
| Marketing | 75.4% | 74.9% | 59.9% | 1000 |
| Materials-Engineering | 48.9% | 44.8% | 35.5% | 1000 |
| Mechanical-Engineering | 43.0% | 39.4% | 32.1% | 1000 |
| Nondestructive-Testing | 47.4% | 45.4% | 36.6% | 1000 |
| Patent | 28.0% | 32.0% | 25.0% | 100 |
| Political-Science-and-Sociology | 52.0% | 47.3% | 38.0% | 300 |
| Psychology | 45.5% | 43.2% | 34.6% | 1000 |
| Public-Safety | 39.8% | 38.1% | 28.9% | 1000 |
| Railway-and-Automotive-Engineering | 40.5% | 38.6% | 31.9% | 1000 |
| Real-Estate | 38.5% | 38.5% | 36.5% | 200 |
| Refrigerating-Machinery | 37.2% | 32.7% | 28.2% | 1000 |
| Social-Welfare | 55.8% | 52.6% | 38.6% | 1000 |
| Taxation | 38.0% | 40.0% | 32.5% | 200 |
| Telecommunications-and-Wireless-Technology | 59.0% | 55.5% | 41.3% | 1000 |
| Korean-History | 33.0% | 27.0% | 33.0% | 100 |
| Math | 23.7% | 24.3% | 23.7% | 300 |

</details>

### BF16 대비 GPTQ가 크게 저하된 과목 (top 5 손실)

| 과목 | BF16 | GPTQ | 차이 | 문항 수 |
|---|---:|---:|---:|---:|
| Health | 64.0% | 55.0% | -9.0%p | 100 |
| Education | 64.0% | 58.0% | -6.0%p | 100 |
| Korean-History | 33.0% | 27.0% | -6.0%p | 100 |
| Management | 53.9% | 49.0% | -4.9%p | 1000 |
| Geomatics | 44.5% | 39.8% | -4.7%p | 1000 |

### BF16 대비 FOEM이 크게 저하된 과목 (top 5 손실)

| 과목 | BF16 | FOEM | 차이 | 문항 수 |
|---|---:|---:|---:|---:|
| Education | 64.0% | 37.0% | -27.0%p | 100 |
| Health | 64.0% | 37.0% | -27.0%p | 100 |
| Information-Technology | 69.1% | 49.1% | -20.0%p | 1000 |
| Computer-Science | 72.0% | 54.1% | -17.9%p | 1000 |
| Telecommunications-and-Wireless-Technology | 59.0% | 41.3% | -17.7%p | 1000 |

- BF16이 GPTQ를 앞서는 과목: **40/45개**, BF16이 FOEM을 앞서는 과목: **43/45개**.

---

## 5. 실제 질의 / 추론 예시

동일한 5-shot 프롬프트로 세 모델에 같은 문항(각 과목 test[0])을 질의하고, " A"~" D" 중 어디에 가장 높은 확률을 두었는지 비교했다.

### 5-1. Korean-History (세 모델 모두 오답, 정답 C에 대한 확신도 비교)

> **질문**: 밑줄 친 '왕'의 재위 기간에 있었던 사실로 옳은 것은? 왕은 노론과 소론, 남인을 두루 등용하였으며 젊은 관료들을 재교육하기 위해 초계문신제를 시행하였다. 또 서얼 출신의 유능한 인사를 규장각 검서관으로 등용하였다.
> - A. 동학이 창시되었다.
> - B. 대전회통이 편찬되었다.
> - C. 신해통공이 시행되었다.
> - D. 홍경래의 난이 발생하였다.
>
> **정답: C** (정조 시대 — 신해통공 시행)

| 모델 | 예측 | A | B | C | D | 정오 |
|---|---|---:|---:|---:|---:|---|
| BF16 원본 | **B** | 9.8% | **49.5%** | 14.2% | 26.5% | ✗ |
| GPTQ 4-bit | **B** | 18.8% | **45.2%** | 14.7% | 21.3% | ✗ |
| FOEM 3-bit | **B** | 17.9% | 29.5% | 23.0% | 29.5% | ✗ |

세 모델 모두 정조(규장각·초계문신제)를 영조·순조 시기 정책과 혼동해 오답(B, "대전회통" — 고종 대 편찬)을 골랐다. BF16이 B에 49.5%로 가장 높은 확신을 보인 반면, FOEM은 B와 D에 확률이 분산되어 정답 C에 상대적으로 높은 23.0%를 배분했다. **3B 모델 자체가 이 문항에서 틀리는 것이며, 양자화로 오답이 생긴 게 아님을 확인할 수 있다.**

### 5-2. Math (세 모델 모두 오답, 무작위에 가까운 분포)

> **질문**: 함수 f(x)=-x³+2x+3에 대하여 직선 y=-x+k와 곡선 y=f(x)의 그래프가 서로 다른 두 점에서 만나도록 하는 모든 양수 k의 합은?
> - A. 6  B. 12  C. 18  D. 36
>
> **정답: A**

| 모델 | 예측 | A | B | C | D | 정오 |
|---|---|---:|---:|---:|---:|---|
| BF16 원본 | **D** | 9.4% | 28.9% | 28.9% | **32.8%** | ✗ |
| GPTQ 4-bit | **B** | 12.6% | **34.2%** | 26.6% | 26.6% | ✗ |
| FOEM 3-bit | **D** | 18.6% | 27.0% | 23.8% | **30.6%** | ✗ |

세 모델 모두 정답(A)에 낮은 확률을 부여하고 B·C·D에 고르게 분산돼 있다 — 베이스 모델(BF16)도 동일하게 틀리므로, 이는 **양자화 손실이 아닌 3B 규모 모델의 수리 추론 한계**에 기인한다.

### 5-3. Computer-Science (세 모델 모두 정답, BF16이 가장 확신)

> **질문**: NTFS 파일시스템에서 부팅과정에서 읽어들이는 부분으로 디스크에 저장되어 있는 파일의 정보DB를 담아 놓은 것은?
> - A. VFAT  B. MFT  C. PROM  D. CMOS
>
> **정답: B**

| 모델 | 예측 | A | B | C | D | 정오 |
|---|---|---:|---:|---:|---:|---|
| BF16 원본 | **B** | 0.1% | **99.2%** | 0.4% | 0.3% | ✓ |
| GPTQ 4-bit | **B** | 0.2% | **97.9%** | 1.4% | 0.6% | ✓ |
| FOEM 3-bit | **B** | 14.5% | **69.3%** | 8.8% | 7.3% | ✓ |

사실 기반(factoid) 문항에서는 세 모델 모두 정답을 맞혔다. BF16(99.2%) > GPTQ(97.9%) > FOEM(69.3%) 순으로 확신도가 낮아지는 패턴이 명확하며, 양자화 수준이 낮을수록(비트 수 감소) 분포가 넓어지는(less sharp) 경향을 보인다.

### 5-4. Marketing (세 모델 모두 오답, 양자화 심화될수록 과확신 심화)

> **질문**: 표본을 통해서 모집단의 성질을 정확히 추론하기 위하여 고려할 사항으로 중요하지 않은 것은?
> - A. 모집단 요소들의 동질성 정도  B. 표본의 크기  C. 표본선정 시기  D. 표본조사 예산
>
> **정답: C**

| 모델 | 예측 | A | B | C | D | 정오 |
|---|---|---:|---:|---:|---:|---|
| BF16 원본 | **D** | 1.3% | 1.0% | 36.9% | **60.8%** | ✗ |
| GPTQ 4-bit | **D** | 1.6% | 1.8% | 28.4% | **68.2%** | ✗ |
| FOEM 3-bit | **D** | 3.7% | 1.3% | 4.8% | **90.2%** | ✗ |

세 모델 모두 오답(D)을 골랐지만, BF16은 정답(C)에 36.9%를 분산해 두었고 GPTQ는 28.4%, FOEM은 4.8%에 불과하다. 즉 **BF16 → GPTQ → FOEM 순서로 오답에 대한 과확신(overconfidence)이 심화**되는 패턴이 뚜렷하다. FOEM 3-bit의 경우 D에 90.2%를 몰아 정답을 사실상 배제했다 — 비트 감소에 따른 확률 분포 왜곡이 누적되는 사례다.

---

## 6. 결론

1. **BF16 원본(46.88%)이 기준선**이며, GPTQ 4-bit는 -2.69%p, FOEM 3-bit는 -11.30%p 차이를 보인다. 비트 수가 더 낮은 FOEM 3-bit의 손실이 더 크며, 이는 PPL 결과(8.72 vs 10.87)와 같은 방향이다.
2. **GPTQ vs FOEM 알고리즘 비교**: GPTQ 4-bit가 FOEM 3-bit보다 micro +8.61%p 높지만, 이는 알고리즘 차이가 아닌 **비트 수 차이(4 vs 3)**가 주된 원인일 가능성이 높다. 공정한 비교를 위해 동일 비트(4-bit FOEM vs 4-bit GPTQ, 3-bit FOEM vs 3-bit GPTQ) 실험이 필요하다.
3. **Math 과목**은 세 모델 모두 무작위 기준선(25%) 수준으로, 양자화 손실보다 **3B 베이스 모델의 수리 추론 한계**가 더 크게 작용한다.
4. **Korean-History**는 BF16(33%)·FOEM(33%)이 GPTQ(27%)를 역전하는 패턴이 실제로 확인됐다. 예시 문항(5-1)에서도 세 모델 모두 동일 오답(B)을 골라 이는 베이스 모델 자체의 한국사 지식 한계이며, GPTQ에서 특정 지식이 더 손실되는 방향으로 작동한 것으로 추정된다 (n=100으로 노이즈 가능성 병존).
5. **후속 실험 제안**: ① 동일 비트 FOEM vs GPTQ 비교; ② Math 과목을 위한 chain-of-thought 프롬프트 실험; ③ Calibration curve (confidence vs accuracy) 분석으로 과확신(overconfident) 오답 패턴 정량화.

---

## 7. 산출물

- `Ministral-3-3B-Instruct-2512-BF16_gptq_4bit/README.md` — `## KMMLU 평가 결과` 섹션 포함.
- `Ministral-3-3B-Instruct-2512-BF16_foem_3bit/README.md` — 동일.
- `pseudo/eval_kmmlu.py` — 독립 실행형 KMMLU 평가 스크립트 (BF16 원본 및 양자화 모델 모두 지원).
- `pseudo/quantize_mistral.py` — `eval_kmmlu_quantized()` 통합.
- `pseudo/KMMLU_REPORT.md` — 이 파일 (3-way 비교, 베이스 모델 포함).
