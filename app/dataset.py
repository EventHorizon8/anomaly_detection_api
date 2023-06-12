import pandas as pd
import torch
from torch.utils.data import TensorDataset


class BaseDataset(TensorDataset):
    """
    Preprocessed dataset
    """
    def __init__(self, data, name='train'):
        self.name = name
        # Select columns and perform pre-processing
        labels = pd.DataFrame(data[["sus"]])
        data = pd.DataFrame(
            data[["processId", "parentProcessId", "userId", "mountNamespace", "eventId", "argsNum", "returnValue"]])
        data["processId"] = data["processId"].map(lambda x: 0 if x in [0, 1, 2] else 1)  # Map to OS/not OS
        data["parentProcessId"] = data["parentProcessId"].map(lambda x: 0 if x in [0, 1, 2] else 1)  # Map to OS/not OS
        data["userId"] = data["userId"].map(lambda x: 0 if x < 1000 else 1)  # Map to OS/not OS
        data["mountNamespace"] = data["mountNamespace"].map(
            lambda x: 0 if x == 4026531840 else 1)  # Map to mount access to mnt/ (all non-OS users) /elsewhere
        data["eventId"] = data["eventId"]  # Keep eventId values (requires knowing max value)
        data["returnValue"] = data["returnValue"].map(
            lambda x: 0 if x == 0 else (1 if x > 0 else 2))  # Map to success/success with value/error
        # Extract values
        self.data = torch.as_tensor(data.values, dtype=torch.int64)
        self.labels = torch.as_tensor(labels.values, dtype=torch.int64)

        super().__init__(self.data, self.labels)