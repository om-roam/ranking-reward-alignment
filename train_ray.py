from ray import tune
from ray.tune import Tuner
from ray.train.torch import TorchTrainer
from ray.tune.schedulers import ASHAScheduler
from ray.train import ScalingConfig

from coral.distributed.ray.reward.train_loop import train_coral_fsdp_tune
from coral.distributed.ray.reward.tunables import get_search_space


def main():
    search_space = get_search_space()

    trainer = TorchTrainer(
        train_loop_per_worker=train_coral_fsdp_tune,
        scaling_config=ScalingConfig(
            num_workers=8,
            use_gpu=True,
        ),
    )

    tuner = Tuner(
        trainer,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            num_samples=24,
            scheduler=ASHAScheduler(
                metric="exact_acc",
                mode="max",
                max_t=1024,
                grace_period=128,
                reduction_factor=2,
            ),
        ),
    )

    results = tuner.fit()
    print(
        "Best result:",
        results.get_best_result(metric="exact_acc", mode="max"),
    )


if __name__ == "__main__":
    main()
