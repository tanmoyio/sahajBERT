import pickle
from typing import Any

import hivemind
import torch
import transformers
from transformers import TrainingArguments

from arguments import CollaborationArguments
from utils import logger, LocalMetrics


class CollaborativeCallback(transformers.TrainerCallback):
    """
    This callback monitors and reports collaborative training progress,
    In case of a catastrophic failure, it can also revert training to a backup
    """
    def __init__(self, dht: hivemind.DHT, optimizer: hivemind.CollaborativeOptimizer,
                 model: torch.nn.Module, local_public_key: bytes, statistics_expiration: float,
                 backup_every_steps: int = CollaborationArguments.backup_every_steps):
        super().__init__()
        self.model = model
        self.dht, self.collaborative_optimizer = dht, optimizer
        self.local_public_key = local_public_key
        self.statistics_expiration = statistics_expiration
        self.last_reported_collaboration_step = -1
        self.samples = 0
        self.steps = 0
        self.loss = 0
        self.total_samples_processed = 0
        self.backup_every_steps = backup_every_steps
        self.backup = self.backup_state()

    def on_train_begin(self, args: TrainingArguments, state: transformers.TrainerState,
                       control: transformers.TrainerControl, **kwargs):
        logger.info('Loading state from peers')
        self.collaborative_optimizer.load_state_from_peers()

    def on_step_end(self, args: TrainingArguments, state: transformers.TrainerState,
                    control: transformers.TrainerControl, **kwargs):
        control.should_log = True
        if not self.params_are_finite():
            logger.warning("Found invalid grads, reloading model from saved state")
            self.restore_from_backup(self.backup)
            return control

        if state.log_history:
            self.loss += state.log_history[-1]['loss']
            self.steps += 1
            if self.collaborative_optimizer.local_step != self.last_reported_collaboration_step:
                self.last_reported_collaboration_step = self.collaborative_optimizer.local_step
                self.total_samples_processed += self.samples
                samples_per_second = self.collaborative_optimizer.performance_ema.samples_per_second
                statistics = LocalMetrics(
                    step=self.collaborative_optimizer.local_step,
                    samples_per_second=samples_per_second,
                    samples_accumulated=self.samples,
                    loss=self.loss,
                    mini_steps=self.steps)
                logger.info(f"Step {self.collaborative_optimizer.local_step}")
                logger.info(f"Your current contribution: {self.total_samples_processed} samples")
                logger.info(f"Performance: {samples_per_second} samples per second.")
                if self.steps:
                    logger.info(f"Local loss: {self.loss / self.steps}")

                self.loss = 0
                self.steps = 0
                if self.collaborative_optimizer.is_synchronized:
                    self.dht.store(key=self.collaborative_optimizer.prefix + "_metrics",
                                   subkey=self.local_public_key, value=statistics.dict(),
                                   expiration_time=hivemind.get_dht_time() + self.statistics_expiration,
                                   return_future=True)

                self.backup = self.backup_state()

        self.samples = self.collaborative_optimizer.local_samples_accumulated

        return control

    @torch.no_grad()
    def params_are_finite(self):
        for param in self.model.parameters():
            if not torch.all(torch.isfinite(param)):
                return False
        return True

    @torch.no_grad()
    def backup_state(self) -> Any:
        return pickle.dumps({'model': self.model.state_dict(),
                             'training': self.collaborative_optimizer.opt.state_dict()})

    @torch.no_grad()
    def restore_from_backup(self, backup):
        state = pickle.loads(backup)
        self.model.load_state_dict(state['model'])
        self.collaborative_optimizer.opt.load_state_dict(state['training'])
