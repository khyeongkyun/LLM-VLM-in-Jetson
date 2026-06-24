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

2. `mse_loss_train_without_memory_issue/modeling_*.py` 에서 `BEST_LAYER` 와 `LAST_PRUNED_LAYER` 를 설정

    ```
    << modeling_opt.py >>
    ...
    BEST_LAYER = 2
    LAST_PRUNED_LAYER = 11

    ```

3. `train.py` 를 통해 Replaced Layer에 대한 Retraining 진행

    ```
    python train.py --model opt --model_name facebook/opt-6.7b
    ```
