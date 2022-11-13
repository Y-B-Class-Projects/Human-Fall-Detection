import numpy as np
import torch.optim as optim
import torchmetrics
import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F


# We have 2 output action classes [fall, not-fall]
TOT_ACTION_CLASSES = 2


# lstm classifier definition
class ActionClassificationLSTM(pl.LightningModule):
    # initialise method
    def __init__(self, input_features, hidden_dim, learning_rate=0.001):
        super().__init__()
        # save hyperparameters
        self.save_hyperparameters()
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_features, hidden_dim, batch_first=True)
        # The linear layer that maps from hidden state space to classes
        self.linear = nn.Linear(hidden_dim, TOT_ACTION_CLASSES)

    def forward(self, x):
        # invoke lstm layer
        lstm_out, (ht, ct) = self.lstm(x)
        # invoke linear layer
        return self.linear(ht[-1])

    def training_step(self, batch, batch_idx):
        # get data and labels from batch
        x, y = batch
        # reduce dimension
        y = torch.squeeze(y)
        # convert to long
        y = y.long()
        # get prediction
        y_pred = self(x)
        # calculate loss
        loss = F.cross_entropy(y_pred, y)
        # get probability score using softmax
        prob = F.softmax(y_pred, dim=1)
        # get the index of the max probability
        pred = prob.data.max(dim=1)[1]
        # calculate accuracy
        acc = torchmetrics.functional.accuracy(pred, y)
        dic = {
            'batch_train_loss': loss,
            'batch_train_acc': acc
        }
        # log the metrics for pytorch lightning progress bar or any other operations
        self.log('batch_train_loss', loss, prog_bar=True)
        self.log('batch_train_acc', acc, prog_bar=True)
        # return loss and dict
        return {'loss': loss, 'result': dic}

    def training_epoch_end(self, training_step_outputs):
        # calculate average training loss end of the epoch
        avg_train_loss = torch.tensor([x['result']['batch_train_loss'] for x in training_step_outputs]).mean()
        # calculate average training accuracy end of the epoch
        avg_train_acc = torch.tensor([x['result']['batch_train_acc'] for x in training_step_outputs]).mean()
        # log the metrics for pytorch lightning progress bar and any further processing
        self.log('train_loss', avg_train_loss, prog_bar=True)
        self.log('train_acc', avg_train_acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        # get data and labels from batch
        x, y = batch
        # reduce dimension
        y = torch.squeeze(y)
        # convert to long
        y = y.long()
        # get prediction
        y_pred = self(x)
        # calculate loss
        loss = F.cross_entropy(y_pred, y)
        # get probability score using softmax
        prob = F.softmax(y_pred, dim=1)
        # get the index of the max probability
        pred = prob.data.max(dim=1)[1]
        # calculate accuracy
        acc = torchmetrics.functional.accuracy(pred, y)
        dic = {
            'batch_val_loss': loss,
            'batch_val_acc': acc
        }
        # log the metrics for pytorch lightning progress bar and any further processing
        self.log('batch_val_loss', loss, prog_bar=True)
        self.log('batch_val_acc', acc, prog_bar=True)
        # return dict
        return dic

    def validation_epoch_end(self, validation_step_outputs):
        # calculate average validation loss end of the epoch
        avg_val_loss = torch.tensor([x['batch_val_loss']
                                     for x in validation_step_outputs]).mean()
        # calculate average validation accuracy end of the epoch
        avg_val_acc = torch.tensor([x['batch_val_acc']
                                    for x in validation_step_outputs]).mean()
        # log the metrics for pytorch lightning progress bar and any further processing
        self.log('val_loss', avg_val_loss, prog_bar=True)
        self.log('val_acc', avg_val_acc, prog_bar=True)

    def configure_optimizers(self):
        # adam optimiser
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        # learning rate reducer scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-15,
                                                         verbose=True)
        # scheduler reduces learning rate based on the value of val_loss metric
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1, "monitor": "val_loss"}}
