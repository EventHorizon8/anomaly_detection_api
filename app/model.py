from pyod.models.iforest import IForest

import pickle
import numpy as np

from app.utils import plot_roc

import settings


class IForestModel(object):

    def __init__(self):
        """Init IsolationForest
        Attributes:
            clf: pyod classifier model
        """
        self.name = 'ifor'
        self.clf = IForest(contamination=settings.SettingsConfig.OUTLIER_FRACTION, random_state=settings.SettingsConfig.RANDOM_STATE)

    def decision_function(self, X):
        return self.clf.decision_function(X)

    def fit(self, X):
        self.clf = self.clf.fit(X)

    def predict_proba(self, X):
        y_proba = self.clf.predict_proba(X)
        return y_proba[:, 1]

    def predict(self, X):
        return self.clf.predict(X)

    def pickle_clf(self, path='data/models/ifor_{}.pkl'.format(settings.SettingsConfig.SEED)):
        """Saves the trained classifier for future use.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.clf, f)
            print("Pickled classifier at {}".format(path))

    def train(self, epoch, dataset):
        train_loss = 0
        # fit the data and tag outliers
        self.fit(dataset.data)
        train_scores = self.decision_function(dataset.data)  # positive distances for inlier, negative for outlier
        train_loss = -1 * np.average(train_scores)  # reverse signage
        return train_loss, self

    def validate(self, epoch, dataset):
        # fit the data and tag outliers
        val_scores = self.decision_function(dataset.data)  # positive distances for inlier, negative for outlier
        val_loss = -1 * np.average(val_scores)  # reverse signage
        return val_loss

    def plot_roc(self, X, y, size_x, size_y):
        """Plot the ROC curve for X_test and y_test.
        """
        plot_roc(self.clf, X, y, size_x, size_y)
