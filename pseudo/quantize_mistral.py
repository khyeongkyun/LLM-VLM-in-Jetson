#!/usr/bin/env python
"""Quantize Mistral-Small-3.1-24B (language backbone) with GPTQModel.

일반 GPTQ 와 FOEM 을 같은 코드로 산출한다. 차이는 QuantizeConfig 에 foem= 인자 유무뿐.

  python quantize_mistral.py --method gptq --bits 4
  python quantize_mistral.py --method foem --bits 4   # alpha=0, beta=0.2 (FOEM w/o GPTAQ)

대상 모델은 멀티모달(Mistral3ForConditionalGeneration)이며, GPTQModel 이 mistral3 를
인식하면 language_model 레이어만 양자화하고 vision tower 는 fp16 으로 둔다.
"""
import argparse
import os

from datasets import load_dataset
from huggingface_hub import snapshot_download

from gptqmodel import GPTQModel, QuantizeConfig

try:
    from gptqmodel import FOEMConfig
except ImportError:  # pragma: no cover - 설치된 버전이 FOEM 미포함이면 즉시 알림
    FOEMConfig = None


def get_calibration(nsamples: int):
    """README 와 동일하게 allenai/c4 영문 일부를 사용. 실패 시 wikitext2 로 폴백."""
    try:
        ds = load_dataset(
            "allenai/c4",
            data_files="en/c4-train.00001-of-01024.json.gz",
            split="train",
        )
        return ds.select(range(nsamples))["text"]
    except Exception as e:  # noqa: BLE001
        print(f"[calib] c4 로드 실패 ({e}); wikitext2 로 폴백")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [t for t in ds["text"] if t.strip()]
        return texts[:nsamples]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["gptq", "foem"], required=True)
    ap.add_argument("--bits", type=int, choices=[3, 4], required=True)
    ap.add_argument(
        "--model", default="mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    )
    ap.add_argument("--nsamples", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--alpha", type=float, default=0.0)  # FOEM w/o GPTAQ
    ap.add_argument("--beta", type=float, default=0.2)
    # transformers 5.x SDPA 마스크의 meta-tensor .item() 버그 회피용. eager 가 안전.
    ap.add_argument("--attn", default="eager", choices=["eager", "sdpa"])
    # disk offload 를 켜면 embed_tokens 출력이 meta tensor 가 되어 입력 캡처가 깨짐.
    # RAM(503G)이 충분하므로 끄고 CPU/GPU 에 materialize 한다.
    ap.add_argument("--offload-disk", action="store_true", default=False)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or (
        f"/workspace/pseudo/quantized/MistralSmall3.1-24B-{args.method}-{args.bits}bit"
    )

    if args.method == "foem" and FOEMConfig is None:
        raise SystemExit(
            "설치된 gptqmodel 에 FOEMConfig 가 없습니다. `pip install -U gptqmodel` 후 재시도."
        )

    # 모델은 캐시에서만 로드(재다운로드 방지). 캘리브레이션 데이터셋은 온라인 허용.
    model_path = snapshot_download(args.model, local_files_only=True)
    print(f"[model] local path: {model_path}")

    qcfg = dict(bits=args.bits, group_size=args.group_size, offload_to_disk=args.offload_disk)
    if args.method == "foem":
        qcfg["foem"] = FOEMConfig(alpha=args.alpha, beta=args.beta, device="auto")
    quant_config = QuantizeConfig(**qcfg)
    print(f"[config] {qcfg}")

    calib = get_calibration(args.nsamples)
    print(f"[calib] {len(calib)} samples")

    model = GPTQModel.load(model_path, quant_config, attn_implementation=args.attn)
    model.quantize(calib, batch_size=args.batch_size)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    model.save(out)
    print(f"[done] saved -> {out}")


if __name__ == "__main__":
    main()
