from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image, ImageOps

from fusion_config import DEFAULT_CONFIG_PATH, config_get, load_config
from methodhub.paths import require_exists
from my_method import RARTPGDiffFusion, _write_debug_log

APP_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = APP_DIR / "demo_sample"
SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _parse_cli_defaults() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    config_args, _ = config_parser.parse_known_args()
    cfg = load_config(config_args.config)

    parser = argparse.ArgumentParser(add_help=False, parents=[config_parser])
    parser.add_argument("--max-rounds", "--max_rounds", type=int, default=int(config_get(cfg, "fusion.max_rounds", 2)))
    parser.add_argument("--resize", type=int, default=int(config_get(cfg, "fusion.resize", 256)))
    parser.add_argument("--device", choices=["cuda", "cpu"], default=str(config_get(cfg, "fusion.device", "cuda")))
    parser.add_argument("--rar-dtype", "--rar_dtype", choices=["bf16", "fp16", "fp32"], default=str(config_get(cfg, "fusion.rar_dtype", "bf16")))
    parser.add_argument("--sampling-mode", "--sampling_mode", choices=["posterior", "sde"], default=str(config_get(cfg, "fusion.sampling_mode", "posterior")))
    parser.add_argument("--stop-on-reject", "--stop_on_reject", action=argparse.BooleanOptionalAction, default=bool(config_get(cfg, "fusion.stop_on_reject", True)))
    parser.add_argument("--extract-assessment-reasoning-hidden", "--extract_assessment_reasoning_hidden", action=argparse.BooleanOptionalAction, default=bool(config_get(cfg, "fusion.extract_assessment_reasoning_hidden", False)))
    parser.add_argument("--assessment-reasoning-max-new-tokens", "--assessment_reasoning_max_new_tokens", type=int, default=int(config_get(cfg, "fusion.assessment_reasoning_max_new_tokens", 256)))
    parser.add_argument("--tpgd-options", "--tpgd_options", default=str(config_get(cfg, "paths.tpgd_options")))
    parser.add_argument("--tpgd-checkpoint", "--tpgd_checkpoint", default=str(config_get(cfg, "paths.tpgd_checkpoint")))
    parser.add_argument("--tpgd-prior", "--tpgd_prior", default=str(config_get(cfg, "paths.tpgd_prior")))
    parser.add_argument("--rar-config", "--rar_config", default=str(config_get(cfg, "paths.rar_config")))
    parser.add_argument("--rar-assessment-config", "--rar_assessment_config", default=str(config_get(cfg, "paths.rar_assessment_config")))
    parser.add_argument("--ui-output-root", "--ui_output_root", default=str(config_get(cfg, "paths.ui_output_root", APP_DIR / "outputs" / "ui")))
    parser.add_argument("--default-example", "--default_example", default=str(config_get(cfg, "ui.default_example", "")))
    parser.add_argument("--max-rounds-slider-max", "--max_rounds_slider_max", type=int, default=int(config_get(cfg, "ui.max_rounds_slider_max", 10)))
    args, _ = parser.parse_known_args()
    args.config_data = cfg
    return args


def _choice_index(options: list[str], value: str) -> int:
    try:
        return options.index(value)
    except ValueError:
        return 0


def _list_examples() -> dict[str, Path]:
    examples: dict[str, Path] = {}
    if SAMPLE_DIR.exists():
        for path in sorted(SAMPLE_DIR.iterdir()):
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                examples[path.stem] = path
    return examples


def _safe_stem(name: str) -> str:
    keep = []
    for char in Path(name).stem:
        keep.append(char if char.isalnum() or char in {"-", "_"} else "_")
    stem = "".join(keep).strip("_")
    return stem or "input"


def _resize_image(image: Image.Image, side: int) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    if side and side > 0:
        image = image.resize((side, side), Image.Resampling.BICUBIC)
    return image


@st.cache_resource(show_spinner=False)
def _load_fusion(
    device: str,
    rar_dtype: str,
    sampling_mode: str,
    tpgd_options: str,
    tpgd_checkpoint: str,
    tpgd_prior: str,
    rar_config: str,
    rar_assessment_config: str,
    stop_on_reject: bool,
    extract_assessment_reasoning_hidden: bool,
    assessment_reasoning_max_new_tokens: int,
) -> RARTPGDiffFusion:
    fusion = RARTPGDiffFusion(
        max_rounds=1,
        device=device,
        rar_config_path=Path(rar_config),
        rar_assessment_config=Path(rar_assessment_config),
        rar_dtype=rar_dtype,
        stop_on_reject=stop_on_reject,
        extract_assessment_reasoning_hidden=extract_assessment_reasoning_hidden,
        assessment_reasoning_max_new_tokens=assessment_reasoning_max_new_tokens,
        tpgdiff_kwargs={
            "options_path": Path(tpgd_options),
            "restoration_checkpoint": Path(tpgd_checkpoint),
            "tpgd_checkpoint": Path(tpgd_prior),
            "sampling_mode": sampling_mode,
        },
    )
    return fusion.load()


def _run_fusion(image: Image.Image, stem: str, settings: dict[str, Any]) -> dict[str, Any]:
    require_exists(Path(settings["tpgd_options"]), "TPGD runtime options")
    require_exists(Path(settings["tpgd_checkpoint"]), "TPGD restoration checkpoint")
    require_exists(Path(settings["tpgd_prior"]), "TPGD prior checkpoint")
    require_exists(Path(settings["rar_config"]), "RAR inference config")
    require_exists(Path(settings["rar_assessment_config"]), "RAR assessment config")

    fusion = _load_fusion(
        settings["device"],
        settings["rar_dtype"],
        settings["sampling_mode"],
        settings["tpgd_options"],
        settings["tpgd_checkpoint"],
        settings["tpgd_prior"],
        settings["rar_config"],
        settings["rar_assessment_config"],
        bool(settings["stop_on_reject"]),
        bool(settings["extract_assessment_reasoning_hidden"]),
        int(settings["assessment_reasoning_max_new_tokens"]),
    )
    fusion.max_rounds = int(settings["max_rounds"])
    fusion.stop_on_reject = bool(settings["stop_on_reject"])
    fusion.enable_assessment_reasoning_hidden = bool(settings["extract_assessment_reasoning_hidden"])
    fusion.assessment_reasoning_max_new_tokens = int(settings["assessment_reasoning_max_new_tokens"])

    output_root = Path(settings["ui_output_root"]).expanduser()
    run_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S") / stem
    run_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = run_dir / "features" if bool(settings["extract_assessment_reasoning_hidden"]) else None

    result, history = fusion.process(image, name=stem, feature_dir=feature_dir)

    input_path = run_dir / f"{stem}_input.png"
    final_path = run_dir / f"{stem}_fusion.png"
    history_path = run_dir / f"{stem}_history.json"
    debug_log_path = run_dir / f"{stem}_debug.log"
    image.save(input_path)
    result.save(final_path)

    stages = [
        {
            "label": "Input",
            "image": image,
            "path": str(input_path),
            "distortion": "",
            "decision": "",
            "accepted": True,
        }
    ]
    records = []
    for step in history:
        candidate_path = run_dir / f"{stem}_round{step.round_index + 1}.png"
        step.candidate.save(candidate_path)
        stages.append(
            {
                "label": f"Round {step.round_index + 1}",
                "image": step.candidate,
                "path": str(candidate_path),
                "distortion": step.distortion,
                "decision": step.quality_decision,
                "accepted": step.accepted,
                "stop_reason": step.stop_reason,
                "assessment_reasoning": step.assessment_reasoning,
                "assessment_reasoning_features": step.assessment_reasoning_features,
            }
        )
        records.append(step.to_record(candidate_path))

    payload = {
        "input_path": str(input_path),
        "final_path": str(final_path),
        "debug_log_path": str(debug_log_path),
        "config_path": settings["config_path"],
        "settings": settings,
        "steps": records,
    }
    history_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    _write_debug_log(
        debug_log_path,
        settings=settings,
        records=records,
        input_path=input_path,
        final_path=final_path,
        history_path=history_path,
    )

    return {
        "stages": stages,
        "final": result,
        "run_dir": str(run_dir),
        "history_path": str(history_path),
        "debug_log_path": str(debug_log_path),
        "final_path": str(final_path),
        "records": records,
    }


def _show_stage(result: dict[str, Any]) -> None:
    stages = result["stages"]
    stage_idx = st.slider(
        "选择 IR 迭代阶段",
        min_value=0,
        max_value=len(stages) - 1,
        value=len(stages) - 1,
        step=1,
    )
    stage = stages[stage_idx]

    left, right = st.columns([1, 1])
    with left:
        st.subheader("当前阶段")
        st.image(stage["image"], caption=stage["label"], width="stretch")
    with right:
        st.subheader("本轮信息")
        st.write(f"阶段：**{stage['label']}**")
        if stage_idx > 0:
            st.write(f"RAR 识别退化：**{stage['distortion']}**")
            st.write(f"RAR 质量比较：**{stage['decision']}**")
            if stage.get("assessment_reasoning"):
                st.write("Assessment Reasoning：")
                st.write(stage["assessment_reasoning"])
            features = stage.get("assessment_reasoning_features") or {}
            if features:
                st.write(f"Reasoning hidden states：`{features.get('feature_path', '')}`")
                st.json({
                    "prefix_hidden": features.get("prefix_hidden"),
                    "generated_hidden": features.get("generated_hidden"),
                    "condition_hidden": features.get("condition_hidden"),
                })
            st.write(f"是否接受：**{stage['accepted']}**")
            st.write(f"停止原因：**{stage.get('stop_reason', '')}**")
        st.write(f"图像路径：`{stage['path']}`")
        st.write(f"最终结果：`{result['final_path']}`")
        st.write(f"历史记录：`{result['history_path']}`")
        st.write(f"调试日志：`{result['debug_log_path']}`")

    if len(stages) > 1:
        st.subheader("迭代总览")
        cols = st.columns(min(len(stages), 4))
        for idx, item in enumerate(stages):
            with cols[idx % len(cols)]:
                st.image(item["image"], caption=item["label"], width="stretch")


def main() -> None:
    defaults = _parse_cli_defaults()
    cfg = defaults.config_data
    cuda_visible = config_get(cfg, "runtime.cuda_visible_devices", None)
    if cuda_visible not in (None, "") and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible)

    st.set_page_config(page_title="TPGD + RAR Fusion", layout="wide", initial_sidebar_state="expanded")
    st.title("TPGD + RAR Fusion Demo")
    st.caption(f"Config: `{defaults.config}`")

    examples = _list_examples()
    example_options = ["Upload"] + list(examples.keys())
    default_index = _choice_index(example_options, defaults.default_example) if defaults.default_example else 0

    st.sidebar.header("Input Options")
    upload_file = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp", "webp"])
    selected_name = st.sidebar.selectbox("Or choose a predefined image", example_options, index=default_index)

    st.sidebar.header("Fusion Settings")
    max_rounds_default = max(1, int(defaults.max_rounds))
    slider_max = max(max_rounds_default, int(defaults.max_rounds_slider_max))
    max_rounds = st.sidebar.slider("Max IR rounds", 1, slider_max, max_rounds_default, 1)
    resize_side = st.sidebar.number_input(
        "Resize square side",
        min_value=0,
        max_value=2048,
        value=max(0, int(defaults.resize)),
        step=64,
    )
    device_options = ["cuda", "cpu"]
    dtype_options = ["bf16", "fp16", "fp32"]
    sampling_options = ["posterior", "sde"]
    device = st.sidebar.selectbox("Device", device_options, index=_choice_index(device_options, defaults.device))
    rar_dtype = st.sidebar.selectbox("RAR dtype", dtype_options, index=_choice_index(dtype_options, defaults.rar_dtype))
    sampling_mode = st.sidebar.selectbox(
        "TPGD sampling mode",
        sampling_options,
        index=_choice_index(sampling_options, defaults.sampling_mode),
    )
    stop_on_reject = st.sidebar.checkbox("Stop when RAR rejects candidate", value=bool(defaults.stop_on_reject))

    with st.sidebar.expander("Feature extraction"):
        extract_assessment_reasoning_hidden = st.checkbox(
            "Extract Assessment Reasoning hidden states",
            value=bool(defaults.extract_assessment_reasoning_hidden),
        )
        assessment_reasoning_max_new_tokens = st.number_input(
            "Reasoning max new tokens",
            min_value=32,
            max_value=512,
            value=int(defaults.assessment_reasoning_max_new_tokens),
            step=32,
        )

    with st.sidebar.expander("Weights and config"):
        tpgd_options = st.text_input("TPGD options", defaults.tpgd_options)
        tpgd_checkpoint = st.text_input("TPGD restoration checkpoint", defaults.tpgd_checkpoint)
        tpgd_prior = st.text_input("TPGD prior checkpoint", defaults.tpgd_prior)
        rar_config = st.text_input("RAR inference config", defaults.rar_config)
        rar_assessment_config = st.text_input("RAR assessment config", defaults.rar_assessment_config)
        ui_output_root = st.text_input("UI output root", defaults.ui_output_root)
        st.caption("修改 GPU 和端口后需要重启 run.bash；侧边栏参数会在本次运行中生效。")

    input_image = None
    stem = "input"
    if selected_name == "Upload" and upload_file is not None:
        stem = _safe_stem(upload_file.name)
        input_image = Image.open(upload_file)
    elif selected_name != "Upload":
        stem = _safe_stem(selected_name)
        input_image = Image.open(examples[selected_name])

    left_col, right_col = st.columns(2)
    with left_col:
        st.subheader("Input Image")
        if input_image is None:
            st.info("请上传图像，或从左侧选择一张示例图。")
        else:
            preview = _resize_image(input_image, int(resize_side))
            st.image(preview, width="stretch")

    run_clicked = st.sidebar.button("Run Fusion", type="primary", disabled=input_image is None)
    if run_clicked and input_image is not None:
        image = _resize_image(input_image, int(resize_side))
        settings = {
            "config_path": str(defaults.config),
            "max_rounds": max_rounds,
            "device": device,
            "rar_dtype": rar_dtype,
            "sampling_mode": sampling_mode,
            "stop_on_reject": stop_on_reject,
            "extract_assessment_reasoning_hidden": extract_assessment_reasoning_hidden,
            "assessment_reasoning_max_new_tokens": int(assessment_reasoning_max_new_tokens),
            "tpgd_options": tpgd_options,
            "tpgd_checkpoint": tpgd_checkpoint,
            "tpgd_prior": tpgd_prior,
            "rar_config": rar_config,
            "rar_assessment_config": rar_assessment_config,
            "ui_output_root": ui_output_root,
        }
        with st.spinner("正在运行 TPGD + RAR，请稍等..."):
            st.session_state["fusion_result"] = _run_fusion(image, stem, settings)

    with right_col:
        st.subheader("Processed Output")
        result = st.session_state.get("fusion_result")
        if result is None:
            st.info("运行后这里会显示最终结果和每一轮 IR 候选。")
        else:
            st.image(result["final"], caption="Final", width="stretch")
            st.write(f"输出目录：`{result['run_dir']}`")
            st.write(f"调试日志：`{result['debug_log_path']}`")

    result = st.session_state.get("fusion_result")
    if result is not None:
        st.divider()
        _show_stage(result)


if __name__ == "__main__":
    main()
