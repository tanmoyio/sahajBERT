#!/usr/bin/env python

import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
from hivemind import SizeAdaptiveCompression, Float16Compression, Uniform8BitQuantization
from torch.utils.data import DataLoader

import transformers
from transformers import set_seed, HfArgumentParser
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.trainer_utils import is_main_process
from transformers import AlbertTokenizerFast
from transformers.trainer import Trainer

import hivemind
import torch_xla.core.xla_model as xm

import callback
from lib import LeanAlbertConfig
from lib.models import LeanAlbertForPreTraining
from lib.training.clipped_lamb import LambWithGradientClipping
from lib.training.multi_tpu import TPUCollaborativeOptimizer
from lib.training.noop import NoOpScheduler, IgnoreGradManipulations
from lib.training.offload import OffloadOptimizer

from data import make_lazy_wikioscar_dataset
from data_collator import AlbertDataCollatorForWholeWordMask
from arguments import CollaborationArguments, DatasetArguments, AlbertTrainingArguments, AveragerArguments
import utils

logger = logging.getLogger(__name__)


def setup_logging(training_args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)


def get_model(training_args, config, tokenizer):
    # Find latest checkpoint in output_dir
    output_dir = Path(training_args.output_dir)
    logger.info(f'Checkpoint dir {output_dir}, contents {list(output_dir.glob("checkpoint*"))}')
    latest_checkpoint_dir = max(output_dir.glob('checkpoint*'), default=None, key=os.path.getctime)

    if latest_checkpoint_dir is not None:
        logger.info(f'Loading model from {latest_checkpoint_dir}')
        model = LeanAlbertForPreTraining.from_pretrained(latest_checkpoint_dir)
    else:
        logger.info(f'Training from scratch')
        model = LeanAlbertForPreTraining(config)
        model.resize_token_embeddings(len(tokenizer))

    return model


def get_optimizer_and_scheduler(training_args, model):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    opt = OffloadOptimizer(
        optimizer_grouped_parameters,
        optim_cls=LambWithGradientClipping,
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
        max_grad_norm=training_args.max_grad_norm,
        clamp_value=training_args.clamp_value,
        debias=True,
    )

    scheduler = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=training_args.total_steps
    )

    return opt, scheduler


class TrainerWithIndependentShuffling(Trainer):
    """
    A version of HuggingFace trainer that shuffles the dataset using a separate random seed.
    Used to ensure that peers don't process batches in the same order.
    """

    def __init__(self, *, data_seed: int, **kwargs):
        super().__init__(**kwargs)
        self.data_seed = data_seed

    def get_train_dataloader(self) -> DataLoader:
        """ Shuffle data independently for each peer to avoid duplicating batches [important for quality] """
        torch.manual_seed(self.data_seed)
        return super().get_train_dataloader()

    def _wrap_model(self, model, training=True):
        return IgnoreGradManipulations(super()._wrap_model(model, training=training))


def main(index: Optional[int] = None):
    tpu = index is not None

    print(f"launched with device index {index}")
    parser = HfArgumentParser((AlbertTrainingArguments, DatasetArguments, CollaborationArguments, AveragerArguments))
    training_args, dataset_args, collaboration_args, averager_args = parser.parse_args_into_dataclasses()

    logger.info(f"Found {len(collaboration_args.initial_peers)} initial peers: {collaboration_args.initial_peers}")
    if len(collaboration_args.initial_peers) == 0:
        raise ValueError("Please specify at least one network endpoint in initial peers.")

    setup_logging(training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config = LeanAlbertConfig.from_pretrained(dataset_args.config_path, cache_dir=dataset_args.cache_dir)
    tokenizer = AlbertTokenizerFast.from_pretrained(dataset_args.tokenizer_path, cache_dir=dataset_args.cache_dir)
    model = get_model(training_args, config, tokenizer)
    model.to(training_args.device)
    if tpu:
        model.tie_weights()

    validators, local_public_key = utils.make_validators(collaboration_args.experiment_prefix)
    if xm.is_master_ordinal():
        dht = hivemind.DHT(
            start=True, initial_peers=collaboration_args.initial_peers,
            client_mode=collaboration_args.client_mode,
            host_maddrs=collaboration_args.host_maddrs,
            announce_maddrs=collaboration_args.announce_maddrs,
            use_ipfs=collaboration_args.use_ipfs,
            record_validators=validators,
        )

        utils.log_visible_maddrs(dht.get_visible_maddrs(), only_p2p=collaboration_args.use_ipfs)
        opt, scheduler = get_optimizer_and_scheduler(training_args, model)
    else:
        dht = None
        opt, scheduler = None, NoOpScheduler()

    total_batch_size_per_step = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    if torch.cuda.device_count() != 0:
        total_batch_size_per_step *= torch.cuda.device_count()

    cctx = xm.CollectiveContext()
    total_batch_size_per_step *= cctx.world_size

    statistics_expiration = collaboration_args.statistics_expiration
    adjusted_target_batch_size = collaboration_args.target_batch_size - collaboration_args.batch_size_lead

    averaging_compression = SizeAdaptiveCompression(
        threshold=2 ** 16 + 1, less=Float16Compression(), greater_equal=Uniform8BitQuantization()),

    collaborative_optimizer = TPUCollaborativeOptimizer(
        opt=opt, dht=dht, scheduler=scheduler, prefix=collaboration_args.experiment_prefix,
        compression=averaging_compression, state_compression=Float16Compression(),
        batch_size_per_step=total_batch_size_per_step, bandwidth=collaboration_args.bandwidth,
        target_batch_size=adjusted_target_batch_size, client_mode=collaboration_args.client_mode,
        reuse_grad_buffers=True, verbose=True, start=True, **asdict(averager_args),
    )

    if xm.is_master_ordinal():
        collaborative_training_callback = callback.CollaborativeCallback(
            dht, collaborative_optimizer, model, local_public_key, statistics_expiration
        )
        callbacks = [collaborative_training_callback]
    else:
        training_args.report_to = 'none'
        callbacks = []

    assert training_args.do_train and not training_args.do_eval
    training_dataset = make_lazy_wikioscar_dataset(tokenizer, shuffle_seed=hash(local_public_key) % 2 ** 31)

    # This data collator will take care of randomly masking the tokens.
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer)
    #AlbertDataCollatorForWholeWordMask(tokenizer=tokenizer, pad_to_multiple_of=training_args.pad_to_multiple_of)

    xm.rendezvous()

    # Note: the code below creates the trainer with dummy scheduler and removes some callbacks.
    # This is done because collaborative training has its own callbacks that take other peers into account.
    trainer = TrainerWithIndependentShuffling(
        model=model, args=training_args, tokenizer=tokenizer,
        data_collator=data_collator, data_seed=hash(local_public_key),
        train_dataset=training_dataset, eval_dataset=None,
        optimizers=(collaborative_optimizer, NoOpScheduler(collaborative_optimizer)),
        callbacks=callbacks
    )
    trainer.remove_callback(transformers.trainer_callback.PrinterCallback)
    trainer.remove_callback(transformers.trainer_callback.ProgressCallback)

    latest_checkpoint_dir = max(Path(training_args.output_dir).glob('checkpoint*'), key=os.path.getctime, default=None)
    trainer.train(model_path=latest_checkpoint_dir)


if __name__ == "__main__":
    main()
