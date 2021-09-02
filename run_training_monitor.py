#!/usr/bin/env python

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import torch
import wandb
from hivemind import Float16Compression, SizeAdaptiveCompression, Uniform8BitQuantization
from hivemind.averaging.training import load_optimizer_state
from transformers import HfArgumentParser, AlbertTokenizerFast

import hivemind
import utils
from arguments import BaseTrainingArguments, CollaborativeOptimizerArguments, AveragerArguments, AlbertTrainingArguments
from run_trainer import get_model, get_optimizer_and_scheduler
from lib.models.config import LeanAlbertConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainingMonitorArguments(BaseTrainingArguments):
    """
    Note: You might want to have several initial peers so that if one dies,
    new workers still can join the collaboration via alive initial peers' addresses.
    Specify initial_peers argument for that purpose
    """

    refresh_period: float = field(default=30, metadata={"help": "Period (in seconds) for fetching the keys from DHT"})
    wandb_project: Optional[str] = field(
        default=None, metadata={"help": "Name of Weights & Biases project to report the training progress to"}
    )
    save_checkpoint_step_interval: int = field(
        default=5, metadata={"help": "Frequency (in steps) of fetching and saving state from peers"}
    )
    model_config_path: str = field(
        default="https://s3.amazonaws.com/models.huggingface.co/bert/albert-large-v2-config.json",
        metadata={"help": "Path to the model config"},
    )
    repo_path: Optional[str] = field(
        default=None, metadata={"help": "Path to local repository to store the model and optimizer states"}
    )
    repo_url: Optional[str] = field(
        default=None, metadata={"help": "URL of Hugging Face Hub repository to upload the model and optimizer states"}
    )
    upload_interval: Optional[float] = field(
        default=None, metadata={"help": "Frequency (in seconds) of uploading the model to Hub"}
    )
    store_checkpoins: bool = field(default=False, metadata={"help": "If True, enables CheckpointHandler"})
    initial_state_path: Optional[str] = field(default=None, metadata={"help": "Path to the initial checkpoint"})
    identity_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a pre-generated private key file. If defined, makes the peer ID deterministic. "
                    "May be generated using ``./p2p-keygen`` from ``go-libp2p-daemon``."
        },
    )
    tokenizer_path: Optional[str] = field(default="tokenizer/tokenizer", metadata={"help": "Path to the tokenizer"})
    cache_dir: Optional[str] = field(default="cache", metadata={"help": "Path to the cache"})


class CheckpointHandler:
    def __init__(
        self,
        monitor_args: TrainingMonitorArguments,
        training_args: AlbertTrainingArguments,
        collab_optimizer_args: CollaborativeOptimizerArguments,
        averager_args: AveragerArguments,
        dht: hivemind.DHT,
    ):
        self.save_checkpoint_step_interval = monitor_args.save_checkpoint_step_interval
        self.repo_path = monitor_args.repo_path
        self.repo_url = monitor_args.repo_url
        self.upload_interval = monitor_args.upload_interval
        self.previous_step = -1

        config = LeanAlbertConfig.from_pretrained(monitor_args.model_config_path)
        tokenizer = AlbertTokenizerFast.from_pretrained(monitor_args.tokenizer_path, cache_dir=monitor_args.cache_dir)
        self.model = get_model(training_args, config, tokenizer)
        opt, scheduler = get_optimizer_and_scheduler(training_args, self.model)
        adjusted_target_batch_size = collab_optimizer_args.target_batch_size - collab_optimizer_args.batch_size_lead

        averaging_compression = SizeAdaptiveCompression(
            threshold=2 ** 16 + 1, less=Float16Compression(), greater_equal=Uniform8BitQuantization()),

        self.collaborative_optimizer = hivemind.CollaborativeOptimizer(
            opt=opt,
            dht=dht,
            prefix=experiment_prefix,
            compression=averaging_compression, state_compression=Float16Compression(),
            bandwidth=collab_optimizer_args.bandwidth,
            target_batch_size=adjusted_target_batch_size,
            client_mode=collab_optimizer_args.client_mode,
            verbose=True,
            start=True,
            **asdict(averager_args),
        )

        if monitor_args.initial_state_path is not None:
            logger.info(f"Loading initial state from {monitor_args.initial_state_path}...")
            parameters_and_extras = [param for param_group in self.collaborative_optimizer.opt.param_groups
                                     for param in param_group["params"]]
            metadata, flat_tensors = torch.load(monitor_args.initial_state_path)
            parameters_and_extras.extend(self.collaborative_optimizer.averager.extra_tensors)
            num_local_tensors = len(parameters_and_extras)
            loaded_parameters_and_extras = flat_tensors[:num_local_tensors]
            loaded_opt_tensors = flat_tensors[num_local_tensors:]
            with torch.no_grad():
                for local_param, loaded_param in zip(parameters_and_extras, loaded_parameters_and_extras):
                    local_param[...] = loaded_param
                load_optimizer_state(
                    self.collaborative_optimizer.opt, metadata["optimizer_metadata"], loaded_opt_tensors)
            self.local_step = max(self.collaborative_optimizer.local_step, metadata["step"])
            logger.info(f"State loaded, starting from step {self.local_step}")

        self.previous_timestamp = time.time()

    def is_time_to_save_state(self, cur_step):
        if self.save_checkpoint_step_interval is None:
            return False
        elif cur_step - self.previous_step >= self.save_checkpoint_step_interval:
            return True
        else:
            return False

    def save_state(self, cur_step):
        logger.info("Saving state from peers")
        self.collaborative_optimizer.load_state_from_peers()
        self.previous_step = cur_step

    def is_time_to_upload(self):
        if self.repo_path is None:
            return False
        elif time.time() - self.previous_timestamp >= self.upload_interval:
            return True
        else:
            return False

    def upload_checkpoint(self, current_loss):
        logger.info("Saving optimizer")
        torch.save(self.collaborative_optimizer.opt.state_dict(), f"{self.repo_path}/optimizer_state.pt")
        self.previous_timestamp = time.time()
        logger.info("Started uploading to Model Hub")
        self.model.push_to_hub(
            repo_name=self.repo_path,
            repo_url=self.repo_url,
            commit_message=f"Step {current_step}, loss {current_loss:.3f}",
        )
        logger.info("Finished uploading to Model Hub")


if __name__ == "__main__":
    parser = HfArgumentParser(
        (TrainingMonitorArguments, AlbertTrainingArguments, CollaborativeOptimizerArguments, AveragerArguments))
    monitor_args, training_args, collab_optimizer_args, averager_args = parser.parse_args_into_dataclasses()

    experiment_prefix = monitor_args.experiment_prefix
    validators, local_public_key = utils.make_validators(experiment_prefix)

    dht = hivemind.DHT(
        start=True,
        initial_peers=monitor_args.initial_peers,
        record_validators=validators,
        use_ipfs=monitor_args.use_ipfs,
        host_maddrs=monitor_args.host_maddrs,
        announce_maddrs=monitor_args.announce_maddrs,
        identity_path=monitor_args.identity_path,
    )
    utils.log_visible_maddrs(dht.get_visible_maddrs(), only_p2p=monitor_args.use_ipfs)

    if monitor_args.wandb_project is not None:
        wandb.init(project=monitor_args.wandb_project)

    current_step = 0
    if monitor_args.store_checkpoins:
        checkpoint_handler = CheckpointHandler(monitor_args, training_args, collab_optimizer_args, averager_args, dht)

    while True:
        metrics_dict = dht.get(experiment_prefix + "_metrics", latest=True)
        if metrics_dict is not None:
            metrics_dict = metrics_dict.value
            metrics = [utils.LocalMetrics.parse_obj(metrics_dict[peer].value) for peer in metrics_dict]
            latest_step = max(item.step for item in metrics)

            if latest_step != current_step:
                logger.debug(f"Got metrics from {len(metrics)} peers")

                for i, metrics_for_peer in enumerate(metrics):
                    logger.debug(f"{i} peer {metrics_for_peer}")

                current_step = latest_step
                alive_peers = 0
                sum_loss = 0
                num_samples = 0
                sum_perf = 0
                sum_mini_steps = 0

                for item in metrics:
                    sum_loss += item.loss
                    alive_peers += 1
                    sum_perf += item.samples_per_second
                    num_samples += item.samples_accumulated
                    sum_mini_steps += item.mini_steps
                current_loss = sum_loss / sum_mini_steps
                logger.info(f"Step #{current_step}\tloss = {current_loss:.5f}")

                if monitor_args.wandb_project is not None:
                    wandb.log(
                        {
                            "loss": current_loss,
                            "alive peers": alive_peers,
                            "samples": num_samples,
                            "performance": sum_perf,
                            "step": latest_step,
                        }
                    )

                if monitor_args.store_checkpoins:
                    if checkpoint_handler.is_time_to_save_state(current_step):
                        checkpoint_handler.save_state(current_step)
                        if checkpoint_handler.is_time_to_upload():
                            checkpoint_handler.upload_checkpoint(current_loss)
        logger.debug("Peer is still alive...")
        time.sleep(monitor_args.refresh_period)
