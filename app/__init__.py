import os

import pandas as pd
import numpy as np
import datetime

from flask import Flask, request
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

from app.dataset import BaseDataset
from app.model import IForestModel


def create_app():
    app = Flask(__name__)
    app.config.from_object('settings.SettingsConfig')

    @app.post('/logs')
    def logs_add_post():
        data = request.json
        print(datetime.datetime.now())
        #dataLogs = data.get('logs')
        filename = os.path.abspath(f"data/raw_logs/labelled_testing_data.csv")
        data_logs = pd.read_csv(filename).sample(100)
        predict_dataset = BaseDataset(data_logs)
        # Load the models
        model = IForestModel()
        model.model_by_pickle()

        # Run model on testing dataset
        outlier_preds = model.predict(predict_dataset.data)
        outlier_score = model.decision_function(predict_dataset.data)
        min_outlier_anomaly_score = np.floor(np.min(outlier_score[np.where(outlier_preds == 1)]) * 10) / 10

        #print(data_logs, outlier_score, min_outlier_anomaly_score, outlier_preds)

        results_with_scores = np.column_stack((data_logs, outlier_score))
        results_with_scores = np.column_stack((results_with_scores, outlier_preds))
        results_with_scores = np.insert(results_with_scores, -1, datetime.datetime.now(), axis=1)
        print(results_with_scores)
        return results_with_scores

    @app.get('/test')
    def logs_test():
        start = datetime.datetime.now()
        filename = os.path.abspath(f"data/raw_logs/labelled_testing_data.csv")
        data_logs = pd.read_csv(filename)
        predict_dataset = BaseDataset(data_logs)
        # Load the models
        model = IForestModel()
        model.model_by_pickle()

        # Run model on testing dataset
        outlier_preds = model.predict(predict_dataset.data)
        outlier_score = model.decision_function(predict_dataset.data)
        min_outlier_anomaly_score = np.floor(np.min(outlier_score[np.where(outlier_preds == 1)]) * 10) / 10
        end = datetime.datetime.now()

        predict_dataset.labels[predict_dataset.labels == 1] = -1
        predict_dataset.labels[predict_dataset.labels == 0] = 1
        metric_tuple = precision_recall_fscore_support(predict_dataset.labels, outlier_preds, average="weighted", pos_label=-1)

        if request.args.get('save') == '1':
            results_with_scores = np.column_stack((data_logs, outlier_score))
            results_with_scores = np.column_stack((results_with_scores, outlier_preds))
            filename = os.path.join("data/stat", f"{model.name}_testing_data_results{datetime.datetime.now().strftime('%Y_%m_%d_%H%M%S')}.csv")
            pd.DataFrame(results_with_scores, columns=list(data_logs.columns) + list(['outlier_score', 'outlier_preds'])).to_csv(filename)


        return {
            "roc_auc_score": roc_auc_score(predict_dataset.labels.numpy(), outlier_preds),
            "time_to_complete": str(end - start),
            "min_outlier_score": min_outlier_anomaly_score,
            "precision": metric_tuple[0],
            "recall": metric_tuple[1],
            "f1-score": metric_tuple[2],
        }

    return app
