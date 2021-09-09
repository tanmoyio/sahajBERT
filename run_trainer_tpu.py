#!/usr/bin/env python
import time
import random
from copy import deepcopy
import multiprocessing as mp

from hivemind import CollaborativeOptimizer

from lib.training.tpu import TPUManager
from run_trainer import *

N_TPUS = 8

import transformers.training_args

transformers.training_args.is_torch_tpu_available = lambda: False  # disable builtin TPU support to use custom code


class TrackableColaborativeOptimizer(CollaborativeOptimizer):
    should_load_params = mp.Event()

    def reset_accumulated_grads_(self):
        self.should_load_params.set()


def main():
    authorizer = authorize_with_huggingface()
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

    # TODO!!!!!!!!!!!!!!!!!!!!!!!!!
    assert training_args.do_train and not training_args.do_eval
    training_dataset = make_lazy_wikioscar_dataset(tokenizer, shuffle_seed=hash(random.random()) % 2 ** 31)

    # This data collator will take care of randomly masking the tokens.
    data_collator = AlbertDataCollatorForWholeWordMask(
        tokenizer=tokenizer, pad_to_multiple_of=training_args.pad_to_multiple_of)

    tpu_manager = TPUManager(deepcopy(model), dataset=training_dataset, collate_fn=data_collator,
                             batch_size=training_args.per_device_train_batch_size,
                             grad_accumulation_steps=training_args.gradient_accumulation_steps,
                             nprocs=N_TPUS, start=True)

    # warmup tpus
    logger.info("Waiting for TPUs to warm up, this may take a minute...")
    tpu_manager.step()
    logger.info("Warmup step 1 / 3 done.")
    tpu_manager.update_model_parameters(model.parameters())
    tpu_manager.step()
    logger.info("Warmup step 2 / 3 done.")
    tpu_manager.step()
    tpu_manager.get_aggregated_gradients()
    tpu_manager.zero_grad()
    logger.info("Warmup step 3 / 3 done.")
    # TODO!!!!!!!!!!!!!!!!!!!!!!!!!

    opt, scheduler = get_optimizer_and_scheduler(training_args, model)

    validators, local_public_key = utils.make_validators(collaboration_args.experiment_prefix)
    dht = hivemind.DHT(
        start=True, initial_peers=collaboration_args.initial_peers,
        client_mode=collaboration_args.client_mode,
        host_maddrs=collaboration_args.host_maddrs,
        announce_maddrs=collaboration_args.announce_maddrs,
        use_ipfs=collaboration_args.use_ipfs,
        record_validators=validators,
        authorizer=authorizer
    )

    utils.log_visible_maddrs(dht.get_visible_maddrs(), only_p2p=collaboration_args.use_ipfs)

    total_batch_size_per_step = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    if torch.cuda.device_count() != 0:
        total_batch_size_per_step *= torch.cuda.device_count()

    statistics_expiration = collaboration_args.statistics_expiration
    adjusted_target_batch_size = collaboration_args.target_batch_size - collaboration_args.batch_size_lead * N_TPUS

    averaging_compression = SizeAdaptiveCompression(
        threshold=2 ** 16 + 1, less=Float16Compression(), greater_equal=Uniform8BitQuantization())

    collaborative_optimizer = TrackableColaborativeOptimizer(
        opt=opt, dht=dht, scheduler=scheduler, prefix=collaboration_args.experiment_prefix,
        compression=averaging_compression, state_compression=Float16Compression(),
        batch_size_per_step=total_batch_size_per_step, bandwidth=collaboration_args.bandwidth,
        target_batch_size=adjusted_target_batch_size, client_mode=collaboration_args.client_mode,
        reuse_grad_buffers=True, verbose=True, start=True, **asdict(averager_args),
    )

    collaborative_training_callback = callback.CollaborativeCallback(
        dht, collaborative_optimizer, model, local_public_key, statistics_expiration
    )

    collaborative_training_callback.on_train_begin(training_args, None, None)
    tpu_manager.update_model_parameters(model.parameters())

    state = transformers.TrainerState()

    while True:
        start_time = time.perf_counter()
        loss, num_accumulated = tpu_manager.step()
        time_delta = time.perf_counter() - start_time
        logger.info(end=f"Accumulated {num_accumulated} gradients at {num_accumulated / time_delta:.3f} g/s\n")

        with torch.no_grad():
            for param, grad_from_tpu in zip(model.parameters(), tpu_manager.get_aggregated_gradients()):
                param.grad.copy_(grad_from_tpu)
            tpu_manager.zero_grad()
            collaborative_optimizer.step()

            if collaborative_optimizer.should_load_params.is_set():
                logger.info("Loading params onto TPU.")
                tpu_manager.update_model_parameters(model.parameters())
                collaborative_optimizer.should_load_params.clear()

        state.log_history.append(dict(loss=loss))
        collaborative_training_callback.on_step_end(training_args, state, None)


if __name__ == "__main__":
    main()
