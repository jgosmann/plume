import behaviors
from sklearn import gaussian_process

global_conf = {
    'duration_in_steps': 200,
    'area': [[-140, 140], [-140, 140], [-80, 0]]
}

predictor = gaussian_process.GaussianProcess(nugget=0.5)
behavior = behaviors.ToMaxVariance(
    margin=10, predictor=predictor, grid_resolution=[15, 15, 9], **global_conf)
