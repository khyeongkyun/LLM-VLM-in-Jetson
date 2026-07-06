from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to prune/retrain.
    Shared by search_pruning_layer.py and replace_and_retrain.py.
    """

    model_name: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
        default="meta-llama/Llama-3.1-8B"
    )

    use_fast: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )


@dataclass
class SearchArguments:
    """
    Arguments for search_pruning_layer.py — locating the [pruning_start_layer,
    pruning_end_layer] range to prune via cosine similarity.
    """

    layer_intervals: Optional[int] = field(
        default=8,
        metadata={"help": "Number of consecutive layers to prune."},
    )

    cosine_num_data: Optional[int] = field(
        default=50,
        metadata={"help": "Amount of data used to calculate cosine similarity."},
    )

    train_num_data: Optional[int] = field(
        default=100000,
        metadata={"help": "Amount of data used to build the dataset the cosine similarity search samples from."},
    )


@dataclass
class TrainingArguments:
    """
    Arguments for replace_and_retrain.py — retraining a single replacement layer
    for a pruned block of layers.
    """

    pruning_start_layer: int = field(
        metadata={"help": "0-indexed first layer to prune (inclusive)."},
    )

    pruning_end_layer: int = field(
        metadata={"help": "0-indexed last layer to prune (inclusive). "
                           "replace_layer approximates layers [pruning_start_layer ... pruning_end_layer]."},
    )

    model: Literal["llama", "opt"] = field(
        default="llama",
        metadata={"help": "Model family to retrain."},
    )

    replace: Literal["none", "mlp", "tf"] = field(
        default="tf",
        metadata={"help": "Replacement layer type."},
    )

    data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Local path for load_from_disk. If None, downloads SlimPajama-6B."},
    )

    output_dir: Optional[str] = field(
        default="./pruned_model",
    )

    batch_size: Optional[int] = field(default=8)

    grad_accum: Optional[int] = field(default=16)

    epochs: Optional[int] = field(default=1)

    eval_every: Optional[int] = field(
        default=500,
        metadata={"help": "Evaluate every N gradient steps."},
    )

    lr: float = field(default=2e-4)
    min_lr: float = field(default=5e-5)
    weight_decay: float = field(default=1e-3)
