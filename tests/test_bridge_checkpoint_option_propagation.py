from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]

FULL_CHECKPOINT_LOADERS = (
    "inference/evaluate_cf.py",
    "inference/generate_all_qa_pairs.py",
    "inference/generate_dpo_multi.py",
    "inference/generate_ecg_answer.py",
    "inference/run_cot_inference.py",
    "inference/tests_multiple_ECG/run_combined_questions.py",
    "inference/tests_multiple_ECG/run_custom_questions.py",
    "projects/rl_finetuning_base.py",
    "scripts/binary_auroc_synonym_eval.py",
    "scripts/compare_psa_vs_original.py",
    "scripts/dpo_pref_eval.py",
    "scripts/echonext_lvef_decoding_comparison.py",
    "scripts/filter_dpo_pairs_by_baseline.py",
    "scripts/generate_dpo_candidates.py",
    "scripts/sample_and_generate_dpo.py",
    "scripts/train_dpo_policy.py",
    "scripts/verify_inference_reproducibility.py",
)

SIDECAR_CHECKPOINT_LOADERS = (
    "inference/evaluate_cf.py",
    "inference/generate_all_qa_pairs.py",
    "inference/generate_dpo_multi.py",
    "inference/generate_ecg_answer.py",
    "inference/tests_multiple_ECG/run_combined_questions.py",
    "inference/tests_multiple_ECG/run_custom_questions.py",
    "projects/rl_finetuning_base.py",
    "scripts/compare_psa_vs_original.py",
    "scripts/dpo_pref_eval.py",
    "scripts/echonext_lvef_decoding_comparison.py",
    "scripts/filter_dpo_pairs_by_baseline.py",
    "scripts/generate_dpo_candidates.py",
    "scripts/sample_and_generate_dpo.py",
    "scripts/train_dpo_policy.py",
    "scripts/verify_inference_reproducibility.py",
)

CONFIG_DATACLASSES = (
    ("utils/config/dpo_finetuning_config.py", "DPOFinetuningConfig"),
    ("utils/config/grpo_finetuning_config.py", "GRPOFinetuningConfig"),
)

DIRECT_STAGE1_LOADERS = (
    ("inference/generate_stage1_val_reports.py", "quantizer"),
    ("inference/generate_stage1_etg_preview.py", "quantizer"),
    ("scripts/probe_bridge.py", "q"),
)

STRUCTURAL_OPTIONS = {"bridge_mix_strategy", "bridge_token_axis"}
STAGE1_CONSTRUCTOR_OPTIONS = {"mix_strategy", "token_axis"}


def _parse(relative_path: str) -> ast.Module:
    path = ROOT / relative_path
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _keyword_map(call: ast.Call) -> dict[str, ast.expr]:
    return {keyword.arg: keyword.value for keyword in call.keywords if keyword.arg}


@pytest.mark.parametrize("relative_path", FULL_CHECKPOINT_LOADERS)
def test_full_checkpoint_loader_forwards_bridge_structure(relative_path: str) -> None:
    tree = _parse(relative_path)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_name(node) == "ECG_Tokenizer_Wrapper"
    ]
    assert calls, f"No ECG_Tokenizer_Wrapper construction found in {relative_path}"

    for call in calls:
        keywords = _keyword_map(call)
        assert STRUCTURAL_OPTIONS <= keywords.keys(), (
            f"{relative_path} must forward {sorted(STRUCTURAL_OPTIONS)} from checkpoint "
            "configuration when reconstructing ECG_Tokenizer_Wrapper"
        )
        for option in STRUCTURAL_OPTIONS:
            expression = ast.unparse(keywords[option])
            assert "config" in expression or "cfg" in expression, (
                f"{relative_path}:{option} must be resolved from checkpoint configuration"
            )


@pytest.mark.parametrize("relative_path", SIDECAR_CHECKPOINT_LOADERS)
def test_sidecar_loader_uses_embedded_checkpoint_structure_as_authority(
    relative_path: str,
) -> None:
    tree = _parse(relative_path)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _call_name(node) == "resolve_checkpoint_bridge_option"
    ]
    resolved_options = {
        call.args[2].value
        for call in calls
        if len(call.args) >= 3
        and isinstance(call.args[2], ast.Constant)
        and isinstance(call.args[2].value, str)
    }

    assert STRUCTURAL_OPTIONS <= resolved_options, (
        f"{relative_path} must resolve both bridge structural options through "
        "the embedded-checkpoint authority helper"
    )


def test_wrapper_validates_structural_tensors_before_permissive_loading() -> None:
    source = (ROOT / "models/ecg_tokenizer_wrapper.py").read_text(encoding="utf-8")
    validation = source.index("validate_bridge_structural_state_dict(state_dict")
    permissive_load = source.index("self.load_state_dict(filtered_state_dict, strict=False)")

    assert validation < permissive_load


@pytest.mark.parametrize(("relative_path", "class_name"), CONFIG_DATACLASSES)
def test_rl_config_declares_bridge_token_axis(
    relative_path: str, class_name: str
) -> None:
    tree = _parse(relative_path)
    config_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    declared = {
        node.target.id
        for node in config_class.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert STRUCTURAL_OPTIONS <= declared, (
        f"{class_name} must accept both checkpoint bridge structural options"
    )


@pytest.mark.parametrize(("relative_path", "quantizer_name"), DIRECT_STAGE1_LOADERS)
def test_direct_stage1_loader_reconstructs_structure_and_attaches_quantizer(
    relative_path: str, quantizer_name: str
) -> None:
    tree = _parse(relative_path)

    constructor_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and STAGE1_CONSTRUCTOR_OPTIONS <= _keyword_map(node).keys()
    ]
    assert constructor_calls, (
        f"{relative_path} must pass mix_strategy and token_axis to its Stage-1 bridge"
    )

    rvq_assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "rvq" for target in node.targets)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "getattr"
        and len(node.value.args) == 3
        and isinstance(node.value.args[0], ast.Name)
        and node.value.args[0].id == quantizer_name
        and isinstance(node.value.args[1], ast.Constant)
        and node.value.args[1].value == "quantizer"
        and isinstance(node.value.args[2], ast.Name)
        and node.value.args[2].id == quantizer_name
    ]
    assert rvq_assignments, (
        f"{relative_path} must unwrap the loaded checkpoint quantizer before attachment"
    )

    guarded_attachments = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test_text = ast.unparse(node.test)
        if "token_axis" not in test_text or "time" not in test_text:
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Call) or _call_name(child) != "attach_quantizer":
                continue
            if child.args and isinstance(child.args[0], ast.Name) and child.args[0].id == "rvq":
                guarded_attachments.append(child)

    assert guarded_attachments, (
        f"{relative_path} must attach the quantizer only for token_axis=time"
    )


def test_probe_passes_loaded_quantizer_to_bridge_loader() -> None:
    tree = _parse("scripts/probe_bridge.py")
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_name(node) == "load_bridge"
    ]
    assert any(
        len(call.args) >= 3
        and isinstance(call.args[2], ast.Name)
        and call.args[2].id == "q"
        for call in calls
    ), "probe_bridge must pass the loaded tokenizer quantizer into load_bridge"


@pytest.mark.parametrize(("relative_path", "_"), DIRECT_STAGE1_LOADERS)
def test_direct_stage1_loader_rejects_any_bridge_key_mismatch(
    relative_path: str, _: str
) -> None:
    source = (ROOT / relative_path).read_text(encoding="utf-8")

    assert ".missing_keys" in source
    assert ".unexpected_keys" in source
    assert "raise RuntimeError" in source


def test_stage1_report_output_accepts_filename_without_parent_directory() -> None:
    source = (ROOT / "inference/generate_stage1_val_reports.py").read_text(
        encoding="utf-8"
    )

    assert 'os.path.dirname(out_path) or "."' in source


def test_probe_uses_strict_shared_waveform_loader() -> None:
    tree = _parse("scripts/probe_bridge.py")
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_name(node) == "load_ecg_waveform"
    ]

    assert calls, "probe_bridge must fail closed through load_ecg_waveform"


def test_rl_sidecar_bridge_structure_survives_checkpoint_reload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class StubBaseProject:
        def __init__(self, config, wandb_wrapper):
            self.config = config
            self.wandb_wrapper = wandb_wrapper

    projects_package = types.ModuleType("projects")
    projects_package.__path__ = []
    base_project_module = types.ModuleType("projects.base_project")
    base_project_module.BaseProject = StubBaseProject
    models_package = types.ModuleType("models")
    models_package.__path__ = []
    wrapper_module = types.ModuleType("models.ecg_tokenizer_wrapper")
    wrapper_module.ECG_Tokenizer_Wrapper = object
    transformers_module = types.ModuleType("transformers")
    transformers_module.__path__ = []
    transformers_module.AutoTokenizer = SimpleNamespace()
    tokenization_utils_module = types.ModuleType("transformers.tokenization_utils")
    files_handler_module = types.ModuleType("utils.files_handler")
    files_handler_module.load_yaml = lambda _: {}
    monkeypatch.setitem(sys.modules, "projects", projects_package)
    monkeypatch.setitem(sys.modules, "projects.base_project", base_project_module)
    monkeypatch.setitem(sys.modules, "models", models_package)
    monkeypatch.setitem(sys.modules, "models.ecg_tokenizer_wrapper", wrapper_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setitem(
        sys.modules,
        "transformers.tokenization_utils",
        tokenization_utils_module,
    )
    monkeypatch.setitem(sys.modules, "utils.files_handler", files_handler_module)

    module_path = ROOT / "projects/rl_finetuning_base.py"
    spec = importlib.util.spec_from_file_location("rl_finetuning_base_under_test", module_path)
    assert spec is not None and spec.loader is not None
    rl_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rl_module)

    class ConcreteProject(rl_module.RLFinetuningProjectBase):
        def _setup_inference_objects(self):
            return {}

        def _setup_training_objects(self):
            return {}

        def _setup_validation_objects(self):
            return {}

        def _setup_test_objects(self):
            return {}

    class FakeTokenizer:
        pad_token = None
        eos_token = "<eos>"
        pad_token_id = None
        eos_token_id = 0

    class FakeModel:
        def _load_state_dict(self, state_dict, strict):
            assert state_dict == {}
            assert strict is False

    wrapper_calls = []
    monkeypatch.setattr(
        rl_module.AutoTokenizer,
        "from_pretrained",
        lambda _: FakeTokenizer(),
        raising=False,
    )
    monkeypatch.setattr(
        rl_module,
        "ECG_Tokenizer_Wrapper",
        lambda **kwargs: wrapper_calls.append(kwargs) or FakeModel(),
    )
    monkeypatch.setattr(
        rl_module,
        "load_yaml",
        lambda _: {
            "bridge_mix_strategy": "concat_linear",
            "bridge_token_axis": "time",
        },
    )

    pretrained = {
        "encoder_name": "encoder",
        "quantizer_name": "quantizer",
        "num_quantizers": 8,
        "codebook_size": 512,
        "tokenizer_name": "tokenizer",
        "decoder_name": "decoder",
        "decoder_mode": "llm",
        "huggingface_model_name": "llm",
        "llm_input_embedding_size": 64,
        "bridge_name": "qformer",
    }
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "config.yaml").write_text("sidecar", encoding="utf-8")
    project = ConcreteProject(SimpleNamespace(), None)
    project._load_checkpoint = lambda _: {
        "config": pretrained,
        "model_state_dict": {},
    }

    _, _, returned_config = project._load_model_from_checkpoint(
        str(source_dir / "checkpoint.pt")
    )

    assert project.config.bridge_mix_strategy == "concat_linear"
    assert project.config.bridge_token_axis == "time"
    assert returned_config.bridge_mix_strategy == "concat_linear"
    assert returned_config.bridge_token_axis == "time"
    assert wrapper_calls[-1]["bridge_mix_strategy"] == "concat_linear"
    assert wrapper_calls[-1]["bridge_token_axis"] == "time"

    saved_config = vars(project.config).copy()
    reloaded_project = ConcreteProject(SimpleNamespace(), None)
    reloaded_project._load_checkpoint = lambda _: {
        "config": saved_config,
        "model_state_dict": {},
    }
    _, _, reloaded_config = reloaded_project._load_model_from_checkpoint(
        str(tmp_path / "reloaded" / "checkpoint.pt")
    )

    assert reloaded_config.bridge_mix_strategy == "concat_linear"
    assert reloaded_config.bridge_token_axis == "time"
    assert wrapper_calls[-1]["bridge_mix_strategy"] == "concat_linear"
    assert wrapper_calls[-1]["bridge_token_axis"] == "time"
