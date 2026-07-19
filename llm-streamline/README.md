# Description

ICLR 2025 에서 Spotlight 되었던 Pruning 기법: **LLM Streamline** 을 적용해볼 수 있는 Repository 입니다. 해당 Paper의 코드 ([Link](https://github.com/ruckbreasoning/llm-streamline)) 를 기반으로 하되 확장성을 고려하여 구조 및 예제 등이 재구성되었습니다.

해당 기법에 대한 자세한 설명은 논문 ([Link](https://arxiv.org/abs/2403.19135)) 혹은 Pseudolab 세미나 발표자료 ([Link](https://github.com/khyeongkyun/plab12-llm-on-jetson/blob/main/hkim-LLM-STREAMLINE-ICLR25-spotlight.pdf)) 를 참고하시기 바랍니다.  

---
🔔 UPDATE 🔔

**2026-07-19:** Pruning 된 OPT 모델의 Benchmark 결과

**2026-06-24:** OPT/Llama 테스트 코드 업로드

## Model & Replacement: Implementation

| LLM / Replacement | None | FFN(MLP) | Transformer |
|---|---|---|---|
| OPT ([Link](https://huggingface.co/facebook/models?search=opt)) | ✅ | ✅ | ✅ |
| LLama-3 ([Link](https://huggingface.co/meta-llama/models?search=llama3)] | ✅ | ✅ | ✅ |
| LLama-2 ([Link](https://huggingface.co/meta-llama/models?search=llama2)] | ✅ | ✅ | ✅ |
| GPT-OSS ([Link](https://huggingface.co/openai/models?search=gpt-oss)): `MoE` | ❌ | ❌ | ❌ |

* 각 모델은 Pruning 된 기존 Layer 의 Input/Output Token에 대한 MSE Loss를 사용하여 Replacement Layer (MLP/TF)의 재학습을 진행하였습니다.
* LLama 모델은 
* LLama-2는 Full MHA을 사용하기 때문에 GQA를 사용하는 LLama-3에 비해 trainin/inference 과정에서 out-of-memory 현상이 발생할 수 있습니다.
* LLM-Streamline 기법은 MOE 기반의 LLM 모델 (e.g., GTP-OSS)에 적용되지 않습니다.


## Dataset for Pruning/Retraining

- [EN] SlimPajama-6B ([Link](https://huggingface.co/datasets/DKYoon/SlimPajama-6B)): ✅

- [KR] WanJuan-Korean ([Link](https://huggingface.co/datasets/opendatalab/WanJuan-Korean)): ⬜

- [KR] kowikitext([Link](https://huggingface.co/datasets/heegyu/kowikitext)): ⬜

## Deployment Test

⬜ TensorRT-LLM Library 사용, Quantization 진행 (e.g., SmoothQuant-INT8 or AWQ-INT4)

- Device: NVIDIA Jetson Orin Nano

- Runtime: TensorRT-LLM Runtime




# Environment Setup

```
conda create -n plab-llm-streamline python=3.10 -y
conda activate plab-llm-streamline
pip install -r requirements.txt
```

아래 Resource를 사용하여 실험이 진행되었습니다.
- CPU: Intel Xeon Gold 6336Y 24C 185W 2.4GHz
- GPU: NVIDIA HGX A100 80GB 500W


# Workflow

1. Pruning Layer Searching : Layer Group 별 In/Out hidden state의 Cosine similarity score 기반 - MSE Loss

    ```
    python search_pruning_layer.py \
    --model_name facebook/opt-6.7b \
    --layer_intervals 8

    ...

    >>> The highest cosine similarity comes from hidden_states 2 and hidden_states 10, with a value of 0.9791
    >>> pruning_layer: [2, 9]
    ```

2. Pruning & Retraining : 해당 모델의 Layer Group을 Lightweight Model (e.g., None / MLP / Transformer) 로 대체한 후, 해당 부분의 IN/Out hidden state 값을 유지할 수 있도록 학습

    ```
    python replace_and_retrain.py \
    --model_name facebook/opt-6.7b \
    --model opt \
    --replace mlp \
    --pruning_start_layer 2 \
    --pruning_end_layer 9 \
    --output_dir /path/to/directory/
    --patience 5 \
    ```

    - Best model만 HuggingFace Checkpoint 형태로 저장
    - 5회 연속으로 Best model이 저장되지 않은 경우 Early stopping
    - 단, `--replace mlp`의 경우 대체 레이어가 표준 decoder layer 형태가 아니므로 `modeling_pruned_<model>.py`가 함께 저장 (`trust_remote_code=True`)
    - 추가로, `opt_prune_from2to9_mlp_eval_log.csv` 파일을 통해 (`global_step`, `eval_loss`, `saved`) 정보 확인가능

3. Model benchmark : Pruning 된 모델을 KMMLU Benchmark를 활용하여 평가

    ```
    python benchmark.py \
        --num_fewshot 1 \
        --model_name "/path/to/directory/opt_prune_from2to9_mlp" \
        --model_label "opt_prune_from2to9_mlp" \
        --replace_type "mlp" \
        --output_dir "/path/to/directory/benchmark_results"
    ```

    *OPT 모델의 한국어에 대한 Tokenizer Overflow 이슈로 인해 1-shot setting으로 바꾸어 평가 진행.

# Evaluation

**Benchmark dataset:** [KMMLU Benchmark](https://huggingface.co/datasets/HAERAE-HUB/KMMLU)


| Model (Layers) | NParams | Size(GB) | STEM | App.Sci | HUMSS | Other | Avg |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **OPT-6.7b (32):**<br>**1-shot**|  |  |  |  |  |  |  |
| Dense (-)  | 6.66B            | 12.4            | 30.11            | 30.13           | 24.16            | 25.69            | 27.58
| 🔁 None (24) | 5.05B<br>(⬇ 24.20%)  | 9.4<br>(⬇ 24.19%) | 20.76<br>(⬇ 31.05%) | 18.38<br>(⬇ 39.00%) | 23.19<br>(⬇ 4.01%) | 14.25<br>(⬇ 44.53%) | 19.13<br>(⬇ 30.64%)
| 🔁 MLP (24+@) | 5.18B<br>(⬇ 22.18%) | 9.65<br>(⬇ 22.18%)  | 28.37<br>(⬇ 5.78%) | 24.64<br>(⬇ 18.22%) | 25.34<br>(⬆ 4.88%)  | 23.84<br>(⬇ 7.20%)  | 25.53<br>(⬇ 7.43%)
| 🔁 TF (24+1) | 0.00% | -<br>(-%) | -<br>(-%) | -<br>(-%) | -<br>(-%) | -<br>(-%) | -<br>(-%) | -<br>(-%) |
|<br>|||||||||
| **Llama-3.1-8B (32):**<br>**5-shot**|  |  |  |  |  |  |
| Dense (-)  |  8.03B            | 14.96           | 42.81           | 38.63           | 41.09           | 41.23           | 40.89
| 🔁 TF | 5.41B<br>(⬇ 32.59%) | 10.08<br>(⬇ 32.62%) | 44.86<br>(⬆ 4.79%) | 41.65<br>(⬆ 7.82%) | 39.01<br>(⬇ 5.06%) | 44.21<br>(⬆ 7.23%) | 42.41<br>(⬆ 3.72%)
|<br>|||||||||
| **Llama-2-7b-hf  (32):**<br>**5-shot**|  |  |  |  |  |  |  |
| Dense (-)  |  6.74B   | 12.55 | 18.79          | 16.67 | 25.50 | 18.47  | 19.79
| 🔁 TF | 0.00% | -<br>(-%) | -<br>(-%) | -<br>(-%) | -<br>(-%) | -<br>(-%) | -<br>(-%) | -<br>(-%) |
|<br>|||||||||

* Llama-2/3 모델의 Benchmark 결과는 저자가 공유한 HuggingFace 모델 ([Llama-2](https://huggingface.co/XiaodongChen/Llama-2-4.7B) / [Llama-3](https://huggingface.co/XiaodongChen/Llama-3.1-5.4B))을 재사용하여 진행하였습니다. 해당 모델은 MSE loss가 아닌, LLM loss (다음 토큰 예측 loss)가 사용되었습니다.

