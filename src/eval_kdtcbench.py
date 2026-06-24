"""K-DTCBench 평가 — 한국어 Document/Table/Chart 4지선다 VQA.

NCSOFT/K-DTCBench (CC-BY-NC-4.0)
- 240문제 (document/table/chart 각 80문제)
- 한국어 네이티브 이미지, 4지선다 MCQA
- 정확매칭 채점 → LLM judge 불필요, 재현 가능

비교 목적:
  fp16 기준선 vs GPTQ-4bit (kocalib) — 양자화 손상을 한국어 VLM 태스크로 측정

실행:
  python src/eval_kdtcbench.py --model fp16
  python src/eval_kdtcbench.py --model gptq --quant-dir models/Llama-3.2-11B-Vision-Instruct-gptq-4bit-kocalib
결과: results/kdtcbench_<model>.json
"""
import argparse
import json
import re
import time
from pathlib import Path

import mllama_compat  # noqa: F401

import torch
from datasets import load_dataset

from config import load_config, hf_token, resolve
from benchmark import load_model

ROOT = Path(__file__).resolve().parent.parent
ORIG = ROOT / "models" / "Llama-3.2-11B-Vision-Instruct"


_PROMPT_TMPL = """\
다음 이미지를 보고 질문에 대한 올바른 답을 A, B, C, D 중 하나만 선택하세요.

질문: {question}

A. {a}
B. {b}
C. {c}
D. {d}

정답 (A/B/C/D 중 하나만):"""


def _build_inputs(processor, image, question: str, choices: dict, device):
    prompt = _PROMPT_TMPL.format(
        question=question,
        a=choices["a"], b=choices["b"], c=choices["c"], d=choices["d"],
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(images=image, text=text, return_tensors="pt")
    return inputs.to(device)


def _extract_answer(text: str) -> str | None:
    """모델 출력에서 A/B/C/D 추출. 첫 등장 알파벳 우선."""
    text = text.strip()
    m = re.search(r"\b([A-D])\b", text.upper())
    return m.group(1) if m else None


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_eval(model, processor, num_samples: int | None = None,
             per_category: int | None = None) -> dict:
    """K-DTCBench 전체 평가. 카테고리별 + 전체 정확도 반환.

    per_category: 지정 시 각 카테고리(document/table/chart)에서 N개씩 균등 추출.
                  데이터셋이 카테고리 순으로 정렬돼 있어 num_samples는 앞쪽만 집어
                  특정 카테고리에 쏠리므로, 공정한 기준선엔 per_category 사용.
    """
    ds = load_dataset("NCSOFT/K-DTCBench", split="test")
    if per_category:
        by_cat: dict[str, list[int]] = {}
        for idx, c in enumerate(ds["category"]):
            by_cat.setdefault(c, []).append(idx)
        sel = [i for c in sorted(by_cat) for i in by_cat[c][:per_category]]
        ds = ds.select(sel)
    elif num_samples:
        ds = ds.select(range(min(num_samples, len(ds))))

    device = _model_device(model)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    cats: dict[str, dict] = {}  # category → {correct, total, latency_sum}
    total_correct, total_n = 0, 0
    t_start = time.time()

    model.eval()
    for i, row in enumerate(ds):
        cat = row["category"]
        if cat not in cats:
            cats[cat] = {"correct": 0, "total": 0, "latency_sum": 0.0}

        choices = {"a": row["choice_a"], "b": row["choice_b"],
                   "c": row["choice_c"], "d": row["choice_d"]}
        inputs = _build_inputs(processor, row["image"], row["question"], choices, device)

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        lat = time.perf_counter() - t0

        in_len = inputs["input_ids"].shape[-1]
        gen_text = processor.decode(out[0][in_len:], skip_special_tokens=True)
        pred = _extract_answer(gen_text)
        correct = int(pred == row["answer"]) if pred else 0

        cats[cat]["correct"] += correct
        cats[cat]["total"] += 1
        cats[cat]["latency_sum"] += lat
        total_correct += correct
        total_n += 1

        if (i + 1) % 20 == 0 or (i + 1) == len(ds):
            elapsed = time.time() - t_start
            acc = total_correct / total_n
            print(f"  [{i+1}/{len(ds)}] acc={acc:.3f}  ({elapsed:.0f}s)", flush=True)

    peak_vram = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0

    by_cat = {
        c: {
            "accuracy": round(v["correct"] / v["total"], 4),
            "correct": v["correct"],
            "total": v["total"],
            "avg_latency_s": round(v["latency_sum"] / v["total"], 3),
        }
        for c, v in cats.items()
    }
    return {
        "total_accuracy": round(total_correct / total_n, 4),
        "total_correct": total_correct,
        "total_n": total_n,
        "by_category": by_cat,
        "peak_vram_gb": round(peak_vram, 2),
        "elapsed_s": round(time.time() - t_start, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["fp16", "gptq"], required=True)
    ap.add_argument("--quant-dir",
                    default=str(ROOT / "models" / "Llama-3.2-11B-Vision-Instruct-gptq-4bit-kocalib"),
                    help="gptq 모델 경로")
    ap.add_argument("--num-samples", type=int, default=None,
                    help="평가 샘플 수 제한 (앞에서부터, None=전체 240)")
    ap.add_argument("--per-category", type=int, default=None,
                    help="카테고리별 N개씩 균등 추출 (공정한 기준선용)")
    ap.add_argument("--out-tag", default="", help="결과 파일명 접미사")
    args = ap.parse_args()

    token = hf_token()
    path = args.quant_dir if args.model == "gptq" else str(ORIG)

    print(f"[load] {args.model} 모델 로드 중… ({path})", flush=True)
    kind = "gptq" if args.model == "gptq" else "mllama_fp16"
    model, processor = load_model(kind, path, token=token)
    print("[load] OK", flush=True)

    print("[eval] K-DTCBench 평가 시작…", flush=True)
    results = run_eval(model, processor, args.num_samples, args.per_category)

    suffix = f"_{args.out_tag}" if args.out_tag else f"_{args.model}"
    out = ROOT / "results" / f"kdtcbench{suffix}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(
        {"model": args.model, "quant_dir": args.quant_dir, **results},
        ensure_ascii=False, indent=2
    ), encoding="utf-8")

    print(f"\n[done] {out}")
    print(f"  전체 정확도: {results['total_accuracy']:.4f} ({results['total_correct']}/{results['total_n']})")
    for cat, r in results["by_category"].items():
        print(f"  {cat}: {r['accuracy']:.4f} ({r['correct']}/{r['total']})")
    print(f"  peak VRAM: {results['peak_vram_gb']}GB  elapsed: {results['elapsed_s']}s")


if __name__ == "__main__":
    main()
