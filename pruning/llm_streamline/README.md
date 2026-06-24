# Description

ICLR 2025 에서 Spotlight 되었던 Pruning 기법: **LLM Streamline** 에 대한 Repository 입니다. Original Repository ([link](https://github.com/ruckbreasoning/llm-streamline)) 기반으로 하되 다른 LLM 모델로의 확장성을 고려하여 재구성되었습니다.

해당 기법에 대한 자세한 설명은 논문 ([link](https://arxiv.org/abs/2403.19135)) 과 세미나 발표자료 ([link](https://github.com/khyeongkyun/plab12-llm-on-jetson/blob/main/hkim-LLM-STREAMLINE-ICLR25-spotlight.pdf)) 를 참고하시기 바랍니다.  

# Environment Setup

```
pip install -r requirements.txt
```

# Workflow

1. `mseloss_entry.py` 를 실행하여 Pruning 할 Layer 확인

    ```
    python mseloss_entry.py \
    --model_name facebook/opt-6.7b \
    --layer_intervals 8 \
    --output_dir /path/to/directory/
    ```

    ```
    The highest cosine similarity comes from hidden_states 2 and hidden_states 11, with a value of 0.9751
    ```

    [!NOTE] Retraining 에 사용할 Dataset의 Hidden State를 RAM에 저장하기 때문에 Memory Issue가 발생할 가능성이 높습니다. 이 경우, 위에서 얻은 Hidden state layer의 정보를 바탕으로 `replace_with_*_no_memory_issue/train.py` 을 통해 Retraining을 진행하시기 바랍니다.

    Q. 기존 `mseloss_entry.py`과 `replace_with_*_no_memory_issue/train.py`의 차이?

    - Model 기준: Replace 를 진행할 Layer만 로드
    - Dataset 기준: 모든 Dataset이 아닌 Batch 기반의 Dataset 로드

2. `replace_with_*_no_memory_issue/train.py` 를 통해 Replaced Layer에 대한 Retraining 진행

    ```
    python train.py --model opt   --start_pruned_layer 3  --last_pruned_layer 11 [--model_name facebook/opt-6.7b] [--data_path <local_dir>]
    ```

    

