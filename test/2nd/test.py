from test_wandb_simple import log_experiment



config={'rmse': 100,
        'param1': 7,
        'param2': 'example_value',
        'param3': 3.01,
        'project': 'test_project'
}
log_experiment(config)
