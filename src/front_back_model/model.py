from torchvision import models
from torch import nn
import torch
import torchmetrics
from torchvision import transforms
import pytorch_lightning as pl
from PIL import Image
from typing import Union


class Squeeze(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = sorted(dims, reverse=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for dim in self.dims:
            x = x.squeeze(dim)
        return x


class FrontBackModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        mobilenet = models.mobilenet_v3_small(pretrained=True)
        mobilenet.eval()
        self.features = mobilenet.features
        self.avgpool = mobilenet.avgpool
        self.squeezer = Squeeze(2, 3)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(576, 200),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(200, 3)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.f1_score = torchmetrics.F1(num_classes=3, average='macro')
        self.preprocess = transforms.ToTensor()
        self._labels = ['enter', 'leave', 'other']

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.squeezer(x)
        x = self.classifier(x)
        return x

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=3e-4)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss = self.criterion(logits, y)
        self.log('train_loss', float(loss))

        preds = torch.softmax(logits, dim=-1).argmax(dim=-1)
        self.f1_score(preds, y)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss = self.criterion(logits, y)
        self.log('val_loss', float(loss))

        preds = torch.softmax(logits, dim=-1).argmax(dim=-1)
        self.f1_score(preds, y)

        return loss

    def training_epoch_end(self, outs):
        self.log('train_f1', self.f1_score.compute())

    def validation_epoch_end(self, outs):
        self.log('val_f1', self.f1_score.compute())

    def predict(self, img: Union[str, Image.Image]):
        if isinstance(img, str):
            img = Image.open(img).convert('RGB')
        tensor = self.preprocess(img).unsqueeze(0)
        logits = self(tensor)
        preds = logits.argmax(dim=-1).item()
        return self._labels[preds]
