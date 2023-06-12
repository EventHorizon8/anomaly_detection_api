import os

import pandas as pd
import numpy as np
import datetime

from flask import Flask, request
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

import settings
from app.dataset import BaseDataset
from app.model import WhitenedBenchmark
from app.utils import prepare_data, prepare_tensor_data


def create_app():
    app = Flask(__name__)
    app.config.from_object('settings.SettingsConfig')

    @app.post('/logs')
    def logs_add_post():
        start = datetime.datetime.now()
        data = request.json
        data_logs = pd.json_normalize([
          {
            "timestamp": 129.050634,
            "processId": 382,
            "threadId": 382,
            "parentProcessId": 1,
            "userId": 101,
            "mountNamespace": 4026532232,
            "processName": "systemd-resolve",
            "hostName": "ip-10-100-1-217",
            "eventId": 41,
            "eventName": "socket",
            "stackAddresses": [
              140159195621643,
              140159192455417,
              94656731598592
            ],
            "argsNum": 3,
            "returnValue": 15,
            "args": "[{'name': 'domain', 'type': 'int', 'value': 'AF_UNIX'}, {'name': 'type', 'type': 'int', 'value': 'SOCK_DGRAM|SOCK_CLOEXEC'}, {'name': 'protocol', 'type': 'int', 'value': 0}]",

          },
          {
            "timestamp": 129.051238,
            "processId": 379,
            "threadId": 379,
            "parentProcessId": 1,
            "userId": 100,
            "mountNamespace": 4026532231,
            "processName": "systemd-network",
            "hostName": "ip-10-100-1-217",
            "eventId": 41,
            "eventName": "socket",
            "stackAddresses": [
              139853228042507,
              93935071185801,
              93935080775184
            ],
            "argsNum": 3,
            "returnValue": 15,
            "args": "[{'name': 'domain', 'type': 'int', 'value': 'AF_UNIX'}, {'name': 'type', 'type': 'int', 'value': 'SOCK_DGRAM|SOCK_CLOEXEC'}, {'name': 'protocol', 'type': 'int', 'value': 0}]",

          },
          {
            "timestamp": 129.051434,
            "processId": 1,
            "threadId": 1,
            "parentProcessId": 0,
            "userId": 0,
            "mountNamespace": 4026531840,
            "processName": "systemd",
            "hostName": "ip-10-100-1-217",
            "eventId": 1005,
            "eventName": "security_file_open",
            "stackAddresses": [
              140362867191588,
              8103505641674584000
            ],
            "argsNum": 4,
            "returnValue": 0,
            "args": "[{'name': 'pathname', 'type': 'const char*', 'value': '/proc/382/cgroup'}, {'name': 'flags', 'type': 'int', 'value': 'O_RDONLY|O_LARGEFILE'}, {'name': 'dev', 'type': 'dev_t', 'value': 5}, {'name': 'inode', 'type': 'unsigned long', 'value': 38584}]",

          }
        ])

        predict_dataset = prepare_tensor_data(prepare_data(data_logs))
        # Load the models
        model = WhitenedBenchmark().model_by_pickle()
        # Run model on testing dataset
        outlier_preds = model.predict(predict_dataset)
        outlier_score = model.decision_function(predict_dataset)
        # Run model on testing dataset
        end = datetime.datetime.now()
        results_with_scores = np.column_stack((data_logs, outlier_preds))

        results_with_scores = np.column_stack((results_with_scores, outlier_score))
        return pd.DataFrame(results_with_scores, columns=list(data_logs.columns) + list(['outlier_preds', 'outlier_anomaly_score'])).to_json(orient="records")

    @app.get('/test')
    def logs_test():
        start = datetime.datetime.now()
        filename = os.path.abspath(f"data/raw_logs/labelled_testing_data.csv")
        data_logs = pd.read_csv(filename)
        predict_dataset = BaseDataset(data_logs)
        # Load the models
        model = WhitenedBenchmark().model_by_pickle()
        # Run model on testing dataset
        outlier_preds = model.predict(predict_dataset.data)
         # Run model on testing dataset
        end = datetime.datetime.now()

        metric_tuple = precision_recall_fscore_support(predict_dataset.labels, outlier_preds, average="weighted", pos_label=1)

        if request.args.get('save') == '1':
            results_with_scores = np.column_stack((data_logs, outlier_preds))
            filename = os.path.join("data/stat", f"testing_data_results{datetime.datetime.now().strftime('%Y_%m_%d_%H%M%S')}.csv")
            pd.DataFrame(results_with_scores, columns=list(data_logs.columns) + list(['outlier_preds'])).to_csv(filename)

        return {
            "roc_auc_score": roc_auc_score(predict_dataset.labels.numpy(), outlier_preds),
            "time_to_complete": str(end - start),
            "precision": metric_tuple[0],
            "recall": metric_tuple[1],
            "f1-score": metric_tuple[2],
        }

    return app
