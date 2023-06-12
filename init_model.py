import os
import pickle

from datetime import datetime

import settings
from app.model import IForestModel
import pandas as pd
import tqdm

from app.dataset import BaseDataset


def init_model():
    start = datetime.now()
    print("Start: ", start)
    model = IForestModel()

    data = pd.read_csv("data/raw_logs/labelled_training_data.csv")
    val_data = pd.read_csv("data/raw_logs/labelled_validation_data.csv")
    train_dataset = BaseDataset(data)
    val_dataset = BaseDataset(val_data)

    ##########################
    # Train & Validate
    ##########################
    train_loss_log, val_loss_log = [], []
    pbar = tqdm.trange(1, settings.SettingsConfig.TRAINING_EPOCHS + 1)
    for epoch in pbar:
        train_loss, model = model.train(epoch, train_dataset)
        pbar.set_description(f"Epoch: {epoch} | Train Loss: {train_loss}")
        train_loss_log.append(train_loss)

        # Validate model
        val_loss = model.validate(epoch, val_dataset)
        pbar.set_description(f"Epoch: {epoch} | Val Loss: {val_loss}")

        # Save best model
        if len(val_loss_log) == 0 or val_loss < min(val_loss_log):
            model.pickle_clf()
        # Early stopping on validation loss
        if len(val_loss_log[-settings.SettingsConfig.TRAINING_EPOCH_STOP:]) >= settings.SettingsConfig.TRAINING_EPOCH_STOP and val_loss >= max(
                val_loss_log[-settings.SettingsConfig.TRAINING_EPOCH_STOP:]):
            print(f"Early stopping at epoch {epoch}")
            break
        val_loss_log.append(val_loss)

    print(f"Min Val Loss: {min(val_loss_log)}")  # Print minimum validation loss

    end = datetime.now()
    print(f"Time to Complete: {end - start}")


if __name__ == "__main__":
    init_model()
