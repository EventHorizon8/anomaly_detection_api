import json
import os

import pandas as pd
import numpy as np
import datetime

from flask import Flask, request
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

import settings
from app.dataset import BaseDataset
from app.model import ModelWrapper
from app.utils import prepare_data, prepare_tensor_data


def create_app():
    app = Flask(__name__)
    app.config.from_object('settings.SettingsConfig')

    @app.post('/logs')
    def logs_add_post():
        start = datetime.datetime.now()
        data = request.json
        data_logs = pd.json_normalize(data.get('logs'))

        predict_dataset = prepare_tensor_data(prepare_data(data_logs))
        # Load the models
        model = ModelWrapper().model_by_pickle()
        # Run model on testing dataset
        outlier_preds = model.predict(predict_dataset)
        outlier_score = model.decision_function(predict_dataset)
        # Run model on testing dataset
        end = datetime.datetime.now()
        results_with_scores = np.column_stack((data_logs, outlier_preds))

        results_with_scores = np.column_stack((results_with_scores, outlier_score))
        result = pd.DataFrame(results_with_scores, columns=list(data_logs.columns) + list(
            ['anomalyDetected', 'outlierAnomalyScore'])).to_json(orient="records")

        return {"logs": json.loads(result)}

    @app.get('/test')
    def logs_test():
        start = datetime.datetime.now()
        filename = os.path.abspath(f"data/raw_logs/labelled_testing_data.csv")
        data_logs = pd.read_csv(filename)
        predict_dataset = BaseDataset(data_logs)
        # Load the models
        model = ModelWrapper().model_by_pickle()
        # Run model on testing dataset
        outlier_preds = model.predict(predict_dataset.data)
        # Run model on testing dataset
        end = datetime.datetime.now()

        metric_tuple = precision_recall_fscore_support(predict_dataset.labels, outlier_preds, average="weighted",
                                                       pos_label=1)

        if request.args.get('save') == '1':
            results_with_scores = np.column_stack((data_logs, outlier_preds))
            filename = os.path.join("data/stat",
                                    f"testing_data_results{datetime.datetime.now().strftime('%Y_%m_%d_%H%M%S')}.csv")
            pd.DataFrame(results_with_scores, columns=list(data_logs.columns) + list(['outlier_preds'])).to_csv(
                filename)

        return {
            "rocAucScore": roc_auc_score(predict_dataset.labels.numpy(), outlier_preds),
            "timeToComplete": str(end - start),
            "precision": metric_tuple[0],
            "recall": metric_tuple[1],
            "f1Score": metric_tuple[2],
        }

    return app
