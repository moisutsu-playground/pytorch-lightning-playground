import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import BertJapaneseTokenizer

from .datasets import TsvDataset


class LivedoorDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers=os.cpu_count(),
        tokenizer=BertJapaneseTokenizer.from_pretrained(
            "cl-tohoku/bert-base-japanese-whole-word-masking"
        ),
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = tokenizer

        self.labels = [
            "it-life-hack",
            "livedoor-homme",
            "peachy",
            "smax",
            "topic-news",
            "dokujo-tsushin",
            "kaden-channel",
            "movie-enter",
            "sports-watch",
        ]

        self.label_dim = len(self.labels)
        self.label2id = {label: _id for _id, label in enumerate(self.labels)}
        self.id2label = {value: key for key, value in self.label2id.items()}

        self.train, self.val, self.test = None, None, None

        self.corpus_dir = Path("corpus/livedoor")

    def setup(self, stage=None):
        self.train, self.val, self.test = (
            TsvDataset(self.corpus_dir / name, self.transform)
            for name in ["train.tsv", "val.tsv", "test.tsv"]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def transform(self, x):
        title, _, label = x
        return (torch.tensor(self.tokenizer.encode(title, max_length=80, padding="max_length")), self.label2id[label])
