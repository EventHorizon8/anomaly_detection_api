from os import environ
import torch
import numpy as np


class SettingsConfig(object):
    SECRET_KEY = environ.get('SECRET_KEY')
    API_KEY = environ.get('API_KEY')
    CUDA_DISABLED = environ.get('CUDA_DISABLED')
    SEED = int(environ.get('SEED', 4))
    RANDOM_STATE = SEED
    TRAINING_EPOCHS = int(environ.get('TRAINING_EPOCHS', 20))
    TRAINING_EPOCH_STOP = int(environ.get('TRAINING_EPOCH_STOP', 15))
    OUTLIER_FRACTION = float(environ.get('OUTLIER_FRACTION', 0.05))
    BATCH_SIZE = int(environ.get('BATCH_SIZE', 128))

    use_cuda = torch.cuda.is_available() and not CUDA_DISABLED
    DEVICE = torch.device('cuda' if use_cuda else 'cpu')

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if use_cuda:
        torch.cuda.manual_seed(SEED)
