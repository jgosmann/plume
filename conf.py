import behaviors
import GPy as gpy
import numpy as np
import prediction

global_conf = {
    'duration_in_steps': 200,
    'area': [[-140, 140], [-140, 140], [-80, 0]]
}

predictor = prediction.GPyAdapter(gpy.kern.rbf(
    input_dim=3, lengthscale=np.sqrt(5)))
behavior = behaviors.DUCB(
    margin=10, predictor=predictor, grid_resolution=[52, 52, 12], kappa=0.15e-9,
    gamma=-1e-18, target_precision=1, **global_conf)
