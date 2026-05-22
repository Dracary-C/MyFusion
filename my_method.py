from __future__ import annotations

import argparse
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import yaml
from PIL import Image

from fusion_config import DEFAULT_CONFIG_PATH, config_get, load_config
from methodhub import build
from methodhub.paths import default_repo_root, push_cwd, push_sys_path, require_exists


DISTORTION_QUERY = (
    "What is the primary ONE degradation observed in the evaluated image? "
    "Answer the question using a single word or phrase."
)
QUALITY_COMPARE_QUERY = (
    "Which of the two images, Image A or Image B, do you consider to be of better quality? "
    "Answer the question using a single word or phrase."
)
ASSESSMENT_REASONING_QUERY = "Please evaluate the image's quality and provide your reasons."

DEFAULT_TPGD_RESTORE_CKPT = Path(
    "/data/chenzt/model_weights/tpgd/universal/ablation-d1-c1-s1/latest_G.pth"
)
DEFAULT_TPGD_PRIOR_CKPT = Path("/data/chenzt/model_weights/tpgd/prior/tpgd_ViT-B-32.pt")


def _first_existing(*paths: Path) -> Optional[Path]:
    for path in paths:
        expanded = path.expanduser()
        if expanded.exists():
            return expanded.resolve()
    return None


def _default_tpgd_options() -> Path:
    return (
        default_repo_root("tpgdiff")
        / "universal-restoration"
        / "config"
        / "tpgd-sde"
        / "options"
        / "test_fast.yml"
    )


def _default_tpgd_prior() -> Optional[Path]:
    return _first_existing(
        DEFAULT_TPGD_PRIOR_CKPT,
        default_repo_root("tpgdiff") / "pretrained" / "tpgd_ViT-B-32.pt",
    )


def _torch_dtype(value: str | torch.dtype) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    key = str(value).lower()
    if key not in aliases:
        raise KeyError(f"Unsupported dtype: {value}")
    return aliases[key]


def _candidate_is_better(answer: str) -> bool:
    text = answer.strip().lower()
    if "image b" in text or text in {"b", "b."} or text.startswith("b "):
        return False
    if "previous" in text and "better" in text:
        return False
    return True


def _tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach()
    stats_tensor = detached.float()
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "mean": float(stats_tensor.mean().cpu()),
        "std": float(stats_tensor.std(unbiased=False).cpu()),
        "min": float(stats_tensor.min().cpu()),
        "max": float(stats_tensor.max().cpu()),
    }


def _feature_tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach()
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
    }


def _write_debug_log(
    log_path: Path,
    *,
    settings: dict[str, Any],
    records: list[dict[str, Any]],
    input_path: Path,
    final_path: Path,
    history_path: Path,
) -> None:
    lines = [
        "MyFusion debug log",
        f"input_path: {input_path}",
        f"final_path: {final_path}",
        f"history_path: {history_path}",
        "",
        "[settings]",
        json.dumps(settings, indent=2, ensure_ascii=False, default=str),
        "",
        "[rounds]",
    ]
    if not records:
        lines.append("no rounds were executed")
    for record in records:
        lines.extend(
            [
                f"round: {record['round_index'] + 1}",
                f"  distortion_query: {DISTORTION_QUERY}",
                f"  distortion_answer: {record['distortion']!r}",
                f"  assessment_reasoning_query: {ASSESSMENT_REASONING_QUERY}",
                f"  assessment_reasoning_answer: {record.get('assessment_reasoning')!r}",
                f"  assessment_reasoning_features: {json.dumps(record.get('assessment_reasoning_features'), ensure_ascii=False, default=str)}",
                f"  tpgd_candidate_path: {record['candidate_path']}",
                f"  quality_query: {QUALITY_COMPARE_QUERY}",
                f"  quality_answer: {record['quality_decision']!r}",
                f"  accepted: {record['accepted']}",
                f"  stop: {record['stop']}",
                f"  stop_reason: {record['stop_reason']}",
                f"  current_latent: {json.dumps(record['current_latent_stats'], ensure_ascii=False)}",
                f"  candidate_latent: {json.dumps(record['candidate_latent_stats'], ensure_ascii=False)}",
                "",
            ]
        )
    log_path.write_text("\n".join(lines), encoding="utf-8")


@dataclass
class HybridStep:
    round_index: int
    distortion: str
    quality_decision: str
    candidate: Image.Image
    accepted: bool
    stop: bool
    stop_reason: str
    current_latent_stats: dict[str, Any]
    candidate_latent_stats: dict[str, Any]
    assessment_reasoning: Optional[str] = None
    assessment_reasoning_features: Optional[dict[str, Any]] = None

    def to_record(self, candidate_path: Path) -> dict[str, Any]:
        record = {
            "round_index": self.round_index,
            "distortion": self.distortion,
            "quality_decision": self.quality_decision,
            "candidate_path": str(candidate_path),
            "accepted": self.accepted,
            "stop": self.stop,
            "stop_reason": self.stop_reason,
            "current_latent_stats": self.current_latent_stats,
            "candidate_latent_stats": self.candidate_latent_stats,
        }
        if self.assessment_reasoning is not None:
            record["assessment_reasoning"] = self.assessment_reasoning
        if self.assessment_reasoning_features is not None:
            record["assessment_reasoning_features"] = self.assessment_reasoning_features
        return record


class RARTPGDiffFusion:
    """Use RAR's latent QA loop to decide whether TPGDiff repair should continue."""

    def __init__(
        self,
        max_rounds: int = 4,
        device: Optional[str] = None,
        rar_root: Optional[Path] = None,
        rar_config_path: Optional[Path] = None,
        rar_assessment_config: Optional[Path] = None,
        rar_dtype: str | torch.dtype = "bf16",
        stop_on_reject: bool = True,
        extract_assessment_reasoning_hidden: bool = False,
        assessment_reasoning_max_new_tokens: int = 256,
        tpgdiff_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        self.max_rounds = max_rounds
        self.stop_on_reject = stop_on_reject
        self.enable_assessment_reasoning_hidden = extract_assessment_reasoning_hidden
        self.assessment_reasoning_max_new_tokens = assessment_reasoning_max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.rar_root = Path(rar_root).resolve() if rar_root else default_repo_root("rar")
        self.rar_config_path = (
            Path(rar_config_path).resolve()
            if rar_config_path
            else self.rar_root / "configs" / "infer_cfg.yaml"
        )
        self.rar_assessment_config = (
            Path(rar_assessment_config).resolve()
            if rar_assessment_config
            else self.rar_root / "iqa" / "config.yaml"
        )
        self.rar_dtype = rar_dtype

        self.tpgdiff_kwargs = dict(tpgdiff_kwargs or {})
        self.tpgdiff_kwargs.setdefault("device", self.device)
        self.tpgdiff_kwargs.setdefault("options_path", _default_tpgd_options())
        self.tpgdiff_kwargs.setdefault("restoration_checkpoint", DEFAULT_TPGD_RESTORE_CKPT)
        prior_ckpt = _default_tpgd_prior()
        if prior_ckpt is not None:
            self.tpgdiff_kwargs.setdefault("tpgd_checkpoint", prior_ckpt)

        self.tpgdiff = build("tpgdiff-runtime", **self.tpgdiff_kwargs)
        self.rar_assessment = build(
            "rar-assessment",
            repo_root=self.rar_root,
            config_path=self.rar_assessment_config,
            device=self.device,
            dtype=self.rar_dtype,
        )
        self.vae = None
        self.image_size = 256
        self.loaded = False

    def load(self) -> "RARTPGDiffFusion":
        self.tpgdiff.load()
        self.rar_assessment.load()
        self._load_rar_vae()
        self.loaded = True
        return self

    def ensure_loaded(self) -> "RARTPGDiffFusion":
        if not self.loaded:
            return self.load()
        return self

    def _load_rar_vae(self) -> None:
        require_exists(self.rar_root, "RAR repo")
        require_exists(self.rar_config_path, "RAR inference config")
        with open(self.rar_config_path, "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)

        self.image_size = int(cfg.get("model", {}).get("image_size", 256))
        vae_path = Path(cfg["vae"]["vae_pretrained"]).expanduser()
        if not vae_path.is_absolute():
            vae_path = self.rar_root / vae_path
        require_exists(vae_path, "RAR SDVAE checkpoint")

        with push_sys_path(self.rar_root), push_cwd(self.rar_root):
            from iqa.sd35 import load_vision_encoder

            vae_dtype = _torch_dtype(cfg["vae"].get("weight_dtype", "float32"))
            self.vae = load_vision_encoder(
                str(self.rar_config_path),
                training=False,
                vision_preprocess={},
                device=self.device,
                dtype=vae_dtype,
            ).to(vae_dtype).eval()

    def _encode_for_rar_qa(self, image: Image.Image) -> torch.Tensor:
        self.ensure_loaded()
        from torchvision import transforms as T

        transform = T.Compose(
            [
                T.Resize((self.image_size, self.image_size)),
                T.CenterCrop(self.image_size),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        vae_dtype = next(self.vae.parameters()).dtype
        lq = transform(image.convert("RGB")).unsqueeze(0).to(self.device, dtype=vae_dtype)
        with torch.no_grad():
            latent = self.vae.encode(lq).to(self.device)
            latent = self.vae.process_in(latent).to(self.device)
        return latent

    def _generate_quality_text(self, inputs: dict[str, Any]) -> str:
        output_texts, _, _, _ = self.rar_assessment.generate(
            inputs,
            latent_input=True,
            save_hidden=False,
        )
        return output_texts[0].replace("\n ", "").strip()

    def assess_distortion(self, latent: torch.Tensor) -> str:
        return self._generate_quality_text(
            {
                "query": [DISTORTION_QUERY],
                "img": latent,
                "img_A": latent,
                "img_B": [None],
                "img_path": ["input"],
                "img_A_path": ["input"],
                "img_B_path": [None],
                "temperature": 0.0,
                "top_p": 0.9,
                "max_new_tokens": 64,
                "task_type": "quality_single_A_noref",
                "output_prob_id": False,
                "output_confidence": False,
            }
        )

    def compare_quality(self, candidate_latent: torch.Tensor, previous_latent: torch.Tensor) -> str:
        return self._generate_quality_text(
            {
                "query": [QUALITY_COMPARE_QUERY],
                "img": [None],
                "img_A": candidate_latent,
                "img_B": previous_latent,
                "img_path": [None],
                "img_A_path": ["candidate"],
                "img_B_path": ["previous"],
                "temperature": 0.0,
                "top_p": 0.9,
                "max_new_tokens": 64,
                "task_type": "quality_compare_noref",
                "output_prob_id": False,
                "output_confidence": False,
            }
        )

    def extract_assessment_reasoning_hidden(
        self,
        latent: torch.Tensor,
        *,
        round_index: int,
        name: str,
        feature_dir: Optional[Path] = None,
    ) -> tuple[str, dict[str, Any]]:
        inputs = {
            "query": [ASSESSMENT_REASONING_QUERY],
            "img": latent,
            "img_A": latent,
            "img_B": [None],
            "img_path": ["input"],
            "img_A_path": ["input"],
            "img_B_path": [None],
            "temperature": 0.0,
            "top_p": 0.9,
            "max_new_tokens": int(self.assessment_reasoning_max_new_tokens),
            "task_type": "quality_single_A_noref",
            "output_prob_id": True,
            "output_confidence": False,
        }
        with torch.no_grad():
            output_texts, _, _, _, generated_hidden, prefix_hidden = self.rar_assessment.generate(
                inputs,
                latent_input=True,
                save_hidden=True,
            )

        answer = output_texts[0].replace("\n ", "").strip()
        condition_hidden = torch.cat([prefix_hidden, generated_hidden], dim=1)
        feature_record: dict[str, Any] = {
            "query": ASSESSMENT_REASONING_QUERY,
            "task_type": "quality_single_A_noref",
            "generated_hidden": _feature_tensor_summary(generated_hidden),
            "prefix_hidden": _feature_tensor_summary(prefix_hidden),
            "condition_hidden": _feature_tensor_summary(condition_hidden),
        }

        if feature_dir is not None:
            feature_dir.mkdir(parents=True, exist_ok=True)
            feature_path = feature_dir / f"{name}_round{round_index + 1}_assessment_reasoning_hidden.pt"
            torch.save(
                {
                    "round_index": round_index,
                    "query": ASSESSMENT_REASONING_QUERY,
                    "answer": answer,
                    "task_type": "quality_single_A_noref",
                    "generated_hidden": generated_hidden.detach().cpu(),
                    "prefix_hidden": prefix_hidden.detach().cpu(),
                    "condition_hidden": condition_hidden.detach().cpu(),
                    "generated_hidden_summary": feature_record["generated_hidden"],
                    "prefix_hidden_summary": feature_record["prefix_hidden"],
                    "condition_hidden_summary": feature_record["condition_hidden"],
                },
                feature_path,
            )
            feature_record["feature_path"] = str(feature_path)

        return answer, feature_record

    def process(
        self,
        image: Image.Image,
        name: str = "sample",
        feature_dir: Optional[Path] = None,
    ) -> tuple[Image.Image, list[HybridStep]]:
        self.ensure_loaded()
        current = image.convert("RGB")
        current_latent = self._encode_for_rar_qa(current)
        history: list[HybridStep] = []

        for round_index in range(self.max_rounds):
            current_latent_stats = _tensor_summary(current_latent)
            distortion = self.assess_distortion(current_latent)
            assessment_reasoning = None
            assessment_reasoning_features = None
            if self.enable_assessment_reasoning_hidden:
                assessment_reasoning, assessment_reasoning_features = self.extract_assessment_reasoning_hidden(
                    current_latent,
                    round_index=round_index,
                    name=name,
                    feature_dir=feature_dir,
                )
            candidate = self.tpgdiff.restore(current)
            candidate_latent = self._encode_for_rar_qa(candidate)
            candidate_latent_stats = _tensor_summary(candidate_latent)
            quality_decision = self.compare_quality(candidate_latent, current_latent)

            accepted = _candidate_is_better(quality_decision)
            reached_max_rounds = round_index == self.max_rounds - 1
            rejected_stop = self.stop_on_reject and not accepted
            stop = rejected_stop or reached_max_rounds
            if rejected_stop:
                stop_reason = "rar_rejected_candidate"
            elif reached_max_rounds:
                stop_reason = "max_rounds_reached"
            else:
                stop_reason = "continue"
            history.append(
                HybridStep(
                    round_index=round_index,
                    distortion=distortion,
                    quality_decision=quality_decision,
                    candidate=candidate,
                    accepted=accepted,
                    stop=stop,
                    stop_reason=stop_reason,
                    current_latent_stats=current_latent_stats,
                    candidate_latent_stats=candidate_latent_stats,
                    assessment_reasoning=assessment_reasoning,
                    assessment_reasoning_features=assessment_reasoning_features,
                )
            )

            if accepted or not self.stop_on_reject:
                current = candidate
                current_latent = candidate_latent
            if stop:
                break

        return current, history


def _parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    config_args, _ = config_parser.parse_known_args()
    cfg = load_config(config_args.config)

    tpgd_root = default_repo_root("tpgdiff")
    rar_root = default_repo_root("rar")
    default_input = Path(config_get(cfg, "paths.input", Path(__file__).resolve().parent / "demo_sample" / "lowlight1.png"))
    default_prior = _default_tpgd_prior()

    parser = argparse.ArgumentParser(
        description="Run the RAR + TPGDiff fusion loop on one image.",
        parents=[config_parser],
    )
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output-dir", type=Path, default=Path(config_get(cfg, "paths.output_dir", Path(__file__).resolve().parent / "outputs" / "tpgd_rar")))
    parser.add_argument("--max-rounds", type=int, default=int(config_get(cfg, "fusion.max_rounds", 2)))
    parser.add_argument("--resize", type=int, default=int(config_get(cfg, "fusion.resize", 256)), help="Resize square side before fusion; use 0 to keep original size.")
    parser.add_argument("--device", default=config_get(cfg, "fusion.device", None))
    parser.add_argument("--rar-dtype", default=str(config_get(cfg, "fusion.rar_dtype", "bf16")), choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--rar-config", type=Path, default=Path(config_get(cfg, "paths.rar_config", rar_root / "configs" / "infer_cfg.yaml")))
    parser.add_argument("--rar-assessment-config", type=Path, default=Path(config_get(cfg, "paths.rar_assessment_config", rar_root / "iqa" / "config.yaml")))
    parser.add_argument("--tpgd-options", type=Path, default=Path(config_get(cfg, "paths.tpgd_options", tpgd_root / "universal-restoration" / "config" / "tpgd-sde" / "options" / "test_fast.yml")))
    parser.add_argument("--tpgd-checkpoint", type=Path, default=Path(config_get(cfg, "paths.tpgd_checkpoint", DEFAULT_TPGD_RESTORE_CKPT)))
    parser.add_argument("--tpgd-prior", type=Path, default=Path(config_get(cfg, "paths.tpgd_prior", default_prior)))
    parser.add_argument("--sampling-mode", default=str(config_get(cfg, "fusion.sampling_mode", "posterior")), choices=["posterior", "sde"])
    parser.add_argument("--stop-on-reject", action=argparse.BooleanOptionalAction, default=bool(config_get(cfg, "fusion.stop_on_reject", True)))
    parser.add_argument("--extract-assessment-reasoning-hidden", action=argparse.BooleanOptionalAction, default=bool(config_get(cfg, "fusion.extract_assessment_reasoning_hidden", False)))
    parser.add_argument("--assessment-reasoning-max-new-tokens", type=int, default=int(config_get(cfg, "fusion.assessment_reasoning_max_new_tokens", 256)))
    parser.add_argument("--cuda-visible-devices", default=config_get(cfg, "runtime.cuda_visible_devices", None))
    parser.add_argument("--tokenizers-parallelism", default=config_get(cfg, "runtime.tokenizers_parallelism", False))
    return parser.parse_args()

def main() -> None:
    args = _parse_args()
    if args.cuda_visible_devices not in (None, ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    if args.tokenizers_parallelism is not None:
        os.environ["TOKENIZERS_PARALLELISM"] = str(args.tokenizers_parallelism).lower()

    input_path = require_exists(args.input, "input image")
    tpgd_options = require_exists(args.tpgd_options, "TPGD runtime options")
    tpgd_checkpoint = require_exists(args.tpgd_checkpoint, "TPGD restoration checkpoint")
    tpgd_prior = require_exists(args.tpgd_prior, "TPGD prior/control checkpoint")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(input_path).convert("RGB")
    if args.resize and args.resize > 0:
        image = image.resize((args.resize, args.resize), Image.Resampling.BICUBIC)

    stem = input_path.stem
    input_copy = args.output_dir / f"{stem}_input.png"
    final_path = args.output_dir / f"{stem}_fusion.png"
    history_path = args.output_dir / f"{stem}_history.json"
    debug_log_path = args.output_dir / f"{stem}_debug.log"
    image.save(input_copy)

    fusion = RARTPGDiffFusion(
        max_rounds=args.max_rounds,
        device=args.device,
        rar_config_path=args.rar_config,
        rar_assessment_config=args.rar_assessment_config,
        rar_dtype=args.rar_dtype,
        stop_on_reject=args.stop_on_reject,
        extract_assessment_reasoning_hidden=args.extract_assessment_reasoning_hidden,
        assessment_reasoning_max_new_tokens=args.assessment_reasoning_max_new_tokens,
        tpgdiff_kwargs={
            "options_path": tpgd_options,
            "restoration_checkpoint": tpgd_checkpoint,
            "tpgd_checkpoint": tpgd_prior,
            "sampling_mode": args.sampling_mode,
        },
    )
    feature_dir = args.output_dir / "features" if args.extract_assessment_reasoning_hidden else None
    result, history = fusion.process(image, name=stem, feature_dir=feature_dir)
    result.save(final_path)

    records = []
    for step in history:
        candidate_path = args.output_dir / f"{stem}_round{step.round_index + 1}.png"
        step.candidate.save(candidate_path)
        records.append(step.to_record(candidate_path))

    run_settings = {
        "config_path": str(args.config),
        "max_rounds": args.max_rounds,
        "stop_on_reject": args.stop_on_reject,
        "extract_assessment_reasoning_hidden": args.extract_assessment_reasoning_hidden,
        "assessment_reasoning_max_new_tokens": args.assessment_reasoning_max_new_tokens,
        "resize": args.resize,
        "device": args.device,
        "rar_dtype": args.rar_dtype,
        "sampling_mode": args.sampling_mode,
        "tpgd_options": str(tpgd_options),
        "tpgd_checkpoint": str(tpgd_checkpoint),
        "tpgd_prior": str(tpgd_prior),
        "rar_config": str(args.rar_config),
        "rar_assessment_config": str(args.rar_assessment_config),
    }
    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "config_path": str(args.config),
                "input_path": str(input_path),
                "input_copy": str(input_copy),
                "final_path": str(final_path),
                "debug_log_path": str(debug_log_path),
                "settings": run_settings,
                "steps": records,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    _write_debug_log(
        debug_log_path,
        settings=run_settings,
        records=records,
        input_path=input_path,
        final_path=final_path,
        history_path=history_path,
    )

    print(f"input: {input_copy}")
    print(f"final: {final_path}")
    print(f"history: {history_path}")
    print(f"debug_log: {debug_log_path}")
    for record in records:
        print(
            f"round {record['round_index'] + 1}: accepted={record['accepted']} "
            f"distortion={record['distortion']!r} decision={record['quality_decision']!r}"
        )


if __name__ == "__main__":
    main()
