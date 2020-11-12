from pathlib import Path
from torch.utils.data import Dataset


class TsvDataset(Dataset):
    def __init__(self, corpus_path: Path, transform=lambda x: x):
        super().__init__()
        self.dataset = []

        with corpus_path.open() as f:
            self.dataset.extend([transform(line.strip().split("\t")) for line in f])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]
