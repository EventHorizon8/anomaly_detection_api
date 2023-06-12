import pandas as pd
import torch
from torch.utils.data import TensorDataset

from app.utils import prepare_data, prepare_tensor_data


class BaseDataset(TensorDataset):
    """
    Preprocessed dataset
    """
    def __init__(self, data, name='train'):
        self.name = name
        # Select columns and perform pre-processing
        labels = pd.DataFrame(data[["sus"]])
        data = prepare_data(data)
        # Extract values
        self.data = prepare_tensor_data(data)
        self.labels = prepare_tensor_data(labels)
        self.columns = ["processId", "parentProcessId", "userId", "mountNamespace", "eventId", "argsNum", "returnValue"]
        self.columns_labels = ["sus"]
        super().__init__(self.data, self.labels)

    def get_input_shape(self):  # note that does not return actual shape, but is used to configure model for categorical data
        num_classes = self.data.max(dim=0)[0] + 1
        num_classes[4] = 1011  # Manually set eventId range as 0-1011 (1010 is max value)
        return num_classes

    def get_distribution(self):
        return 'categorical'

