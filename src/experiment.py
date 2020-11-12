from pytorch_lightning.core import datamodule
import torch
import pytorch_lightning as pl

from src.data import LivedoorDataModule
from src.models import BertForSentenceClassification

class Experiment(pl.LightningModule):
    def __init__(self, dataset_name: str, batch_size: int, learning_rate: float):
        super().__init__()

        self.learning_rate = learning_rate

        if dataset_name == "livedoor":
            self.datamodule = LivedoorDataModule(batch_size)
        else:
            raise ValueError(f"No such a dataset name {dataset_name}")

        self.model = BertForSentenceClassification(self.datamodule.label_dim)

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def forward(self, batch):
        out = self.model(batch)
        return out

    def training_step(self, batch, batch_idx):
        word_ids, labels = batch
        loss = self.model.fit(word_ids, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        word_ids, labels = batch
        loss = self.model.fit(word_ids, labels)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def save(self, trainer: pl.Trainer, save_path: str):
        trainer.save_checkpoint(save_path)

    def fit(self, trainer: pl.Trainer):
        trainer.fit(self, datamodule=self.datamodule)
