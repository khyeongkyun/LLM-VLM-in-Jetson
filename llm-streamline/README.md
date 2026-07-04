# Description

ICLR 2025 에서 Spotlight 되었던 Pruning 기법: **LLM Streamline** 을 적용해볼 수 있는 Repository 입니다. 해당 Paper의 코드 ([Link](https://github.com/ruckbreasoning/llm-streamline)) 를 기반으로 하되 확장성을 고려하여 구조 및 예제 등이 재구성되었습니다.

해당 기법에 대한 자세한 설명은 논문 ([Link](https://arxiv.org/abs/2403.19135)) 혹은 Pseudolab 세미나 발표자료 ([Link](https://github.com/khyeongkyun/plab12-llm-on-jetson/blob/main/hkim-LLM-STREAMLINE-ICLR25-spotlight.pdf)) 를 참고하시기 바랍니다.  

---
✅: Worked / ⬜: Not-worked

**Model & Replacement**

| LLM / Replacement | None | FFN(MLP) | Transformer |
|---|---|---|---|
| LLama-2 ([Link](https://huggingface.co/meta-llama/models?search=llama2)] | ⬜ | ✅ | ✅ |
| LLama-3 ([Link](https://huggingface.co/meta-llama/models?search=llama3)] | ⬜ | ✅ | ✅ |
| OPT ([Link](https://huggingface.co/facebook/models?search=opt)) | ⬜ | ✅ | ✅ |
| GPT-OSS ([Link](https://huggingface.co/openai/models?search=gpt-oss)) | ⬜ | ⬜ | ⬜ |

**Dataset - Pruning/Retraining**

- [EN] SlimPajama-6B ([Link](https://huggingface.co/datasets/DKYoon/SlimPajama-6B)): ✅

- [KR] WanJuan-Korean ([Link](https://huggingface.co/datasets/opendatalab/WanJuan-Korean)): ⬜

- [KR] kowikitext([Link](https://huggingface.co/datasets/heegyu/kowikitext)): ⬜

---
🔔 UPDATE 🔔

**2026-00-00:** TBDs


# Environment Setup

```
conda create -n plab-llm-streamline python=3.10 -y
conda activate plab-llm-streamline
pip install -r requirements.txt
```

**System Resource**
- CPU: Intel Xeon Gold 6336Y 24C 185W 2.4GHz
- GPU: NVIDIA HGX A100 80GB 500W


# Workflow

1. Pruning Layer Searching : Layer Group 별 In/Out hidden state의 Cosine similarity score 기반

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
    ```

# Evaluation

LLM : [KMMLU Benchmark](https://huggingface.co/datasets/HAERAE-HUB/KMMLU)


**Replacement: None**
| Model | STEM | Applied Science | HUMSS | Other | **Average** |
|:---:|:---:|:---:|:---:|:---:|:---:|
| LLama-2-7b: 8 layer | - (-%) | - (-%) | - (-%) | - (-%) | **- (-%)** |
| LLama-3-8b: 8 layer | - (-%) | - (-%) | - (-%) | - (-%) | **- (-%)** |
| OPT-6.7b: 8 layer | - (-%) | - (-%) | - (-%) | - (-%) | **- (-%)** |

**Replacement: FFN**
| Model | STEM | Applied Science | HUMSS | Other | **Average** |
|:---:|:---:|:---:|:---:|:---:|:---:|
| LLama-2-7b: 8 layer | - (-%) | - (-%) | - (-%) | - (-%) | **- (-%)** |
| LLama-3-8b: 8 layer | - (-%) | - (-%) | - (-%) | - (-%) | **- (-%)** |
| OPT-6.7b: 8 layer | - (-%) | - (-%) | - (-%) | - (-%) | **- (-%)** |

**Replacement: Transformer**
| Model | STEM | Applied Science | HUMSS | Other | **Average** |
|:---:|:---:|:---:|:---:|:---:|:---:|
| LLama-2-7b: 8 layer | - (-%) | - (-%) | - (-%) | - (-%) | **- (-%)** |
| LLama-3-8b: 8 layer | - (-%) | - (-%) | - (-%) | - (-%) | **- (-%)** |
| OPT-6.7b: 8 layer | - (-%) | - (-%) | - (-%) | - (-%) | **- (-%)** |

