import behaviors

global_conf = {
    'duration_in_steps': 200,
    'area': [[-140, 140], [-140, 140], [-80, 0]]
}

behavior = behaviors.ToMaxVariance(
    margin=10, **global_conf)
