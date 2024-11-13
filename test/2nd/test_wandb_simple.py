import wandb

def log_experiment(run_config):
    # Initialize W&B run
    if run_config.get('project'):
        wandb.init(project=run_config.get('project'), config=run_config)
    else:
        wandb.init(project="ml4_simple", config=run_config)
    # Access configuration parameters
    # wandb_config = wandb.config
    # Log additional config values as metrics (if desired)
    for key, value in run_config.items():
        wandb.log({key: value})
    # Finish the W&B run
    wandb.finish()



def main():
    # Example usage with config parameters
    config = {
        "param1": 5,
        "param2": "example_value",
        "param3": 0.01
    }
    log_experiment(config)

if __name__ == "__main__":
    main()

    