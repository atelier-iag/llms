import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


script = Path(__file__).resolve().parents[1] / "baseline/olmo3/mini/train.py"
spec = importlib.util.spec_from_file_location("mini_train", script)
mini = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mini)


@pytest.fixture
def launch(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(sys, "argv", [str(script), "--train-single", "--save-folder", "runs/test"])
    trainer = Mock(no_checkpoints=False)
    trainer.maybe_load_checkpoint.return_value = False
    saver = Mock(spec=mini.ConfigSaverCallback)
    trainer.callbacks = {"config_saver": saver}
    config = SimpleNamespace(
        init_seed=12536,
        model=Mock(),
        train_module=Mock(dp_config=object(), tp_config=object()),
        dataset=Mock(),
        data_loader=Mock(),
        trainer=Mock(),
        load_path=None,
        as_config_dict=Mock(return_value={"single": True}),
    )
    config.trainer.build.return_value = trainer
    build_config = Mock(return_value=config)
    setup, teardown, distributed, seed = Mock(), Mock(), Mock(), Mock()
    monkeypatch.setattr(mini, "build_config", build_config)
    monkeypatch.setattr(mini, "prepare_training_environment", setup)
    monkeypatch.setattr(mini, "teardown_training_environment", teardown)
    monkeypatch.setattr(mini, "distributed_main", distributed)
    monkeypatch.setattr(mini, "seed_all", seed)
    return SimpleNamespace(**locals())


def test_single_gpu_does_not_initialize_distributed_training(launch):
    mini.main()
    launch.setup.assert_called_once_with(backend=None, shared_filesystem=True)
    launch.distributed.assert_not_called()
    assert launch.config.train_module.dp_config is None
    assert launch.config.train_module.tp_config is None
    assert launch.build_config.call_args.args[0].work_dir == "runs/test"
    launch.seed.assert_called_once_with(12536)
    assert launch.saver.config == {"single": True}
    launch.trainer.fit.assert_called_once_with()
    launch.teardown.assert_called_once_with()


@pytest.mark.parametrize("existing_checkpoint", [False, True])
def test_existing_checkpoint_takes_precedence_over_load_path(launch, existing_checkpoint):
    launch.config.load_path = "runs/previous/step1"
    launch.trainer.maybe_load_checkpoint.return_value = existing_checkpoint
    mini.main()
    if existing_checkpoint:
        launch.trainer.load_checkpoint.assert_not_called()
    else:
        launch.trainer.load_checkpoint.assert_called_once_with(
            "runs/previous/step1", load_trainer_state=False
        )


def test_teardown_runs_when_training_fails(launch):
    launch.trainer.fit.side_effect = RuntimeError("training failed")
    with pytest.raises(RuntimeError, match="training failed"):
        mini.main()
    launch.teardown.assert_called_once_with()


def test_dry_run_reports_single_gpu_config_without_training(launch, monkeypatch):
    monkeypatch.setattr(sys, "argv", sys.argv + ["--dry-run", "trainer.max_duration.value=2"])
    monkeypatch.setattr(mini, "prepare_cli_environment", Mock())
    display = Mock()
    monkeypatch.setattr(mini.rich, "print", display)
    mini.main()
    assert launch.build_config.call_args.args[1] == ["trainer.max_duration.value=2"]
    assert launch.config.train_module.dp_config is None
    display.assert_called_once_with(launch.config)
    launch.setup.assert_not_called()
    launch.trainer.fit.assert_not_called()


def test_distributed_launch_keeps_upstream_entrypoint(launch, monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(script), "--save-folder", "runs/test"])
    mini.main()
    launch.distributed.assert_called_once()
    assert launch.distributed.call_args.args == (launch.build_config,)
    launch.setup.assert_not_called()


def test_single_gpu_rejects_multiple_processes(launch, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(SystemExit) as error:
        mini.main()
    assert error.value.code == 2
    launch.setup.assert_not_called()
