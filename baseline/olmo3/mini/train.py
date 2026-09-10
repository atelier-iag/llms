"""
Mini OLMo 3 baseline derived from the official OLMo-3-1025-7B pre-training script.
State restored to the configuration used around the first machine crash.
"""

import argparse
import os
from pathlib import Path
from typing import List

import rich

from olmo_core.config import DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    NumpyPaddedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.io import is_url
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import CosWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.script_utils import ExperimentConfig, get_cli_parser
from olmo_core.script_utils import main as distributed_main
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    CometCallback,
    ConfigSaverCallback,
    LMEvaluatorCallbackConfig,
    MonkeyPatcherCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import prepare_cli_environment, seed_all

DEFAULT_SEQUENCE_LENGTH = 1024
GLOBAL_BATCH_SIZE = 1024
LR = 3e-4


def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    sequence_length = opts.sequence_length or DEFAULT_SEQUENCE_LENGTH
    tokenizer_config = TokenizerConfig.dolma2()

    model_config = TransformerConfig.olmo3_60M(
        vocab_size=tokenizer_config.padded_vocab_size(),
        attn_backend=AttentionBackendName.torch,
    )

    data_path = Path(__file__).parent / "data" / "train.npy"
    val_data_path = Path(__file__).parent / "data" / "val.npy"

    dataset_config = NumpyFSLDatasetConfig(
        paths=[str(data_path)],
        sequence_length=sequence_length,
        tokenizer=tokenizer_config,
        work_dir=opts.work_dir,
    )

    val_dataset_config = NumpyPaddedFSLDatasetConfig(
        paths=[str(val_data_path)],
        metadata=[{"label": "mini-validation"}],
        sequence_length=sequence_length,
        tokenizer=tokenizer_config,
        work_dir=opts.work_dir,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=34521,
        num_workers=0,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=sequence_length,
        max_sequence_length=sequence_length,
        optim=SkipStepAdamWConfig(
            lr=LR,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(
                    params=["embeddings.weight"],
                    opts=dict(weight_decay=0.0),
                )
            ],
        ),
        scheduler=CosWithWarmup(warmup_steps=5),
        compile_model=False,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
    )

    trainer_config = (
        TrainerConfig(
            save_folder=opts.save_folder,
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=Duration.steps(100),
            hard_stop=Duration.steps(int(597046)),
        )
        .with_callback("monkey_patcher", MonkeyPatcherCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=None,
                save_async=False,
            ),
        )
        .with_callback(
            "comet",
            CometCallback(
                name=opts.name,
                cancel_check_interval=10,
                enabled=False,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=opts.name,
                cancel_check_interval=10,
                enabled=False,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "lm_evaluator",
            LMEvaluatorCallbackConfig(
                eval_dataset=val_dataset_config,
                eval_interval=100,
                eval_duration=Duration.steps(50),
            ),
        )
    )

    return ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
    ).merge(overrides)


def main() -> None:
    parser = get_cli_parser()
    opts, overrides = parser.parse_known_args()
    if not opts.train_single:
        distributed_main(build_config, parser=parser)
        return

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("--train-single requires one process; launch with python")

    if opts.work_dir is None:
        opts.work_dir = "/tmp/olmo-core/dataset-cache" if is_url(opts.save_folder) else opts.save_folder

    config = build_config(opts, overrides)
    config.train_module.dp_config = None
    config.train_module.tp_config = None
    if opts.dry_run:
        prepare_cli_environment()
        rich.print(config)
        return

    # The pinned upstream launcher still initializes NCCL with --train-single.
    # GPU execution works without a process group; loss reduction then stays local.
    prepare_training_environment(backend=None, shared_filesystem=not is_url(opts.save_folder))
    try:
        seed_all(config.init_seed)
        model = config.model.build(init_device="meta")
        train_module = config.train_module.build(model)
        dataset = config.dataset.build()
        data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
        trainer = config.trainer.build(train_module, data_loader)

        for callback in trainer.callbacks.values():
            if isinstance(callback, ConfigSaverCallback):
                callback.config = config.as_config_dict()
                break

        if not trainer.no_checkpoints and not trainer.maybe_load_checkpoint() and config.load_path:
            trainer.load_checkpoint(config.load_path, load_trainer_state=False)

        trainer.fit()
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
