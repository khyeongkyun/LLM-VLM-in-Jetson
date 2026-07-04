"""
awq/pipeline.py

AWQ 양자화 파이프라인을 하나의 클래스로 통합합니다.
모델에 종속되지 않고, HuggingFace AutoModelForCausalLM을 지원하는
모든 모델에 범용적으로 사용할 수 있습니다.
"""

import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

from .calibration import run_calibration, get_calib_dataset
from .quantize import quantize_model
from .export import export_awq_model


# 모델 아키텍처별 양자화에서 제외할 레이어 패턴
# 모든 모델에서 lm_head는 기본 제외 (vocab projection)
DEFAULT_SKIP_PATTERNS = {"lm_head"}


class AWQQuantizer:
    """
    AWQ INT4 양자화 파이프라인.

    HuggingFace CausalLM 모델이면 아키텍처에 관계없이 사용 가능합니다.
    (Qwen, LLaMA, Mistral, Gemma, Phi 등)

    Args:
        model_name: HuggingFace 모델 ID 또는 로컬 경로
        w_bit: 양자화 비트 수 (기본 4)
        group_size: 그룹 양자화 크기 (기본 128)
        zero_point: asymmetric 양자화 사용 여부 (기본 True)
        skip_layers: 양자화에서 제외할 레이어 이름 set (기본: {"lm_head"})
        device_map: 모델 로드 시 device_map (기본 "auto")
        trust_remote_code: trust_remote_code 설정 (기본 True)

    Example:
        >>> quantizer = AWQQuantizer("Qwen/Qwen3-4B")
        >>> quantizer.quantize(calib_data="pileval", output_dir="./output")

        >>> # 이미 로드된 모델 사용
        >>> quantizer = AWQQuantizer.from_pretrained(model, tokenizer)
        >>> quantizer.quantize(calib_data="kowikitext", output_dir="./output")
    """

    def __init__(
        self,
        model_name: str,
        w_bit: int = 4,
        group_size: int = 128,
        zero_point: bool = True,
        skip_layers: set[str] | None = None,
        device_map: str = "auto",
        trust_remote_code: bool = True,
    ):
        self.model_name = model_name
        self.w_bit = w_bit
        self.group_size = group_size
        self.zero_point = zero_point
        self.skip_layers = skip_layers or DEFAULT_SKIP_PATTERNS
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code

        self._model = None
        self._tokenizer = None

    @classmethod
    def from_pretrained(
        cls,
        model: torch.nn.Module,
        tokenizer,
        w_bit: int = 4,
        group_size: int = 128,
        zero_point: bool = True,
        skip_layers: set[str] | None = None,
    ) -> "AWQQuantizer":
        """이미 로드된 모델과 토크나이저로 AWQQuantizer를 생성합니다."""
        instance = cls.__new__(cls)
        instance.model_name = getattr(model.config, "_name_or_path", "unknown")
        instance.w_bit = w_bit
        instance.group_size = group_size
        instance.zero_point = zero_point
        instance.skip_layers = skip_layers or DEFAULT_SKIP_PATTERNS
        instance.device_map = "auto"
        instance.trust_remote_code = True
        instance._model = model
        instance._tokenizer = tokenizer
        return instance

    @property
    def model(self) -> torch.nn.Module:
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer

    def _load_model(self):
        """모델과 토크나이저를 로드합니다."""
        print(f"모델 로드 중: {self.model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=self.trust_remote_code,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device_map,
            trust_remote_code=self.trust_remote_code,
        )
        print(f"모델 로드 완료. Device: {next(self._model.parameters()).device}")

    def _build_config(self, calib_data: str, n_samples: int, seq_len: int) -> dict:
        return {
            "calibration": {
                "dataset": calib_data,
                "n_samples": n_samples,
                "seq_len": seq_len,
            },
            "awq": {
                "w_bit": self.w_bit,
                "group_size": self.group_size,
                "zero_point": self.zero_point,
                "skip_layers": self.skip_layers,
            },
        }

    def quantize(
        self,
        calib_data: str = "pileval",
        output_dir: str | None = None,
        n_samples: int = 128,
        seq_len: int = 512,
    ) -> str:
        """
        전체 AWQ 파이프라인을 실행합니다: calibration → quantize → export.

        Args:
            calib_data: calibration 데이터셋 ("pileval", "wikitext2", "kowikitext", "c4")
            output_dir: 출력 디렉토리 (None이면 자동 생성)
            n_samples: calibration 샘플 수
            seq_len: calibration 시퀀스 길이

        Returns:
            출력 디렉토리 경로
        """
        if output_dir is None:
            short_name = self.model_name.split("/")[-1].lower()
            output_dir = f"./outputs/{short_name}-awq-{calib_data}"

        config = self._build_config(calib_data, n_samples, seq_len)

        print("=" * 60)
        print(f"  AWQ Quantization Pipeline")
        print(f"  모델: {self.model_name}")
        print(f"  Calibration: {calib_data} ({n_samples} samples, seq_len={seq_len})")
        print(f"  양자화: INT{self.w_bit}, group_size={self.group_size}")
        print(f"  스킵 레이어: {self.skip_layers}")
        print(f"  출력: {output_dir}")
        print("=" * 60)

        # Step 1: Calibration
        print(f"\n[1/3] Calibration ({calib_data})...")
        act_stats = run_calibration(self.model, self.tokenizer, config)
        print(f"  {len(act_stats)}개 레이어 통계 수집 완료")

        # Step 2: Quantize
        print(f"\n[2/3] AWQ 양자화 적용 중...")
        model, quant_results = quantize_model(self.model, act_stats, config)
        self._model = model
        print(f"  양자화 완료 ({len(quant_results)}개 레이어)")

        # Step 3: Export
        print(f"\n[3/3] 모델 저장 중...")
        export_awq_model(model, quant_results, self.tokenizer, output_dir, config)

        print("\n" + "=" * 60)
        print(f"  완료! 저장 위치: {output_dir}")
        print("=" * 60)

        return output_dir
