from ray import tune

def get_search_space():
    search_space = {
        "training": {
            "batch_size": tune.choice([16, 32]),
            "num_layers": tune.choice([27, 30]),
        },
        "optimizer": {
            "param_groups": {
                "encoder": {"lr": tune.loguniform(1e-5, 5e-4)},
                "head": {"lr": tune.loguniform(1e-5, 5e-4)},
            }
        },
        "loss": {
            "gamma": tune.choice([1.5, 2.0, 2.5]),
            "mono_coef": tune.uniform(0.05, 0.2),
            "gap_coef": tune.uniform(0.01, 0.1),
            "scale_coef": tune.uniform(0.01, 0.05),
        },
    }
    return search_space