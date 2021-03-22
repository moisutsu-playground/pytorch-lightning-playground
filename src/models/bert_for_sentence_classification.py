from torch import nn
from transformers import BertModel


class BertForSentenceClassification(nn.Module):
    BERT_EMBEDDINGS_DIM = 768

    def __init__(self, label_dim: int):
        super().__init__()
        self.label_dim = label_dim
        self.bert = BertModel.from_pretrained(
            "cl-tohoku/bert-base-japanese-whole-word-masking"
        )
        self.cls2label = nn.Linear(self.BERT_EMBEDDINGS_DIM, label_dim)
        self.criterion = nn.CrossEntropyLoss()

    @staticmethod
    def _make_attention_mask(batch):
        return (batch != 0).float()

    def forward(self, batch):
        attention_mask = self._make_attention_mask(batch)
        cls = self.bert(batch, attention_mask=attention_mask)[1]
        pred_labels = self.cls2label(cls)
        return pred_labels.argmax(dim=1)

    def fit(self, batch, labels):
        attention_mask = self._make_attention_mask(batch)
        cls = self.bert(batch, attention_mask=attention_mask)[1]
        pred_labels = self.cls2label(cls).view(-1, self.label_dim)
        loss = self.criterion(pred_labels, labels)
        return loss
