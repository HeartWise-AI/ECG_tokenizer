import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_main_module(monkeypatch):
    utils_pkg = types.ModuleType("utils")
    utils_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "utils", utils_pkg)

    seed_mod = types.ModuleType("utils.seed")
    seed_mod.set_seed = lambda seed: None
    monkeypatch.setitem(sys.modules, "utils.seed", seed_mod)

    class StubDistributedUtils:
        @staticmethod
        def ddp_setup(gpu_id, world_size):
            pass

        @staticmethod
        def sync_process_group(world_size, device_ids):
            pass

        @staticmethod
        def ddp_cleanup():
            pass

    ddp_mod = types.ModuleType("utils.ddp")
    ddp_mod.DistributedUtils = StubDistributedUtils
    monkeypatch.setitem(sys.modules, "utils.ddp", ddp_mod)

    parser_mod = types.ModuleType("utils.parser")
    parser_mod.HeartWiseParser = object
    monkeypatch.setitem(sys.modules, "utils.parser", parser_mod)

    class StubProjectRegistry:
        @staticmethod
        def get(name):
            raise AssertionError("ProjectRegistry.get should not be reached")

    registry_mod = types.ModuleType("utils.registry")
    registry_mod.ProjectRegistry = StubProjectRegistry
    monkeypatch.setitem(sys.modules, "utils.registry", registry_mod)

    projects_pkg = types.ModuleType("projects")
    projects_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "projects", projects_pkg)

    project_types_mod = types.ModuleType("projects.types")
    project_types_mod.ProjectT = object
    monkeypatch.setitem(sys.modules, "projects.types", project_types_mod)

    class StubWandbWrapper:
        def __init__(self, *args, **kwargs):
            pass

        def finish(self):
            pass

    wandb_mod = types.ModuleType("utils.wandb_wrapper")
    wandb_mod.WandbWrapper = StubWandbWrapper
    monkeypatch.setitem(sys.modules, "utils.wandb_wrapper", wandb_mod)

    config_pkg = types.ModuleType("utils.config")
    config_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "utils.config", config_pkg)

    config_mod = types.ModuleType("utils.config.heartwise_config")
    config_mod.HeartWiseConfig = object
    monkeypatch.setitem(sys.modules, "utils.config.heartwise_config", config_mod)

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "main.py"
    spec = importlib.util.spec_from_file_location("script_main", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_startup_failure_preserves_original_exception(monkeypatch):
    module = _load_main_module(monkeypatch)
    cleanup_calls = []

    def fail_setup(gpu_id, world_size):
        raise RuntimeError("ddp setup failed")

    monkeypatch.setattr(module.DistributedUtils, "ddp_setup", fail_setup)
    monkeypatch.setattr(module.DistributedUtils, "ddp_cleanup", lambda: cleanup_calls.append(True))

    config = SimpleNamespace(
        device=0,
        world_size=1,
        use_wandb=False,
        is_ref_device=True,
        pipeline_project="unused",
    )

    with pytest.raises(RuntimeError, match="ddp setup failed"):
        module.main(config)

    assert cleanup_calls == [True]
