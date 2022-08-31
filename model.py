import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig
import os

ROOT_PATH = os.path.abspath(os.curdir)


class MultiTaskClassify(nn.Module):

    def __init__(self):
        super(MultiTaskClassify, self).__init__()
        PHOBERT_CONFIG = RobertaConfig.from_pretrained(
            ROOT_PATH + "/dataset/phobert/config.json",
            from_tf = False,
            output_hidden_states=False
        )
        self.phobert = RobertaModel.from_pretrained(
            ROOT_PATH + "/dataset/phobert/pytorch_model.bin",
            config = PHOBERT_CONFIG
        )

        self.output_layer = nn.ModuleList()
        for _ in range(6):
            self.output_layer.append(
                nn.Sequential(
                    nn.Linear(768, 128, bias=True),
                    nn.Linear(128, 6, bias=True)
                )
            )
        

    def forward(self, X):
        X_mask = torch.where(X != 1, 1, 0)
        hidden_state, y = self.phobert(input_ids=X, attention_mask=X_mask, return_dict=False)

        y_pred = torch.Tensor()
        y_pred.requires_grad_(True)
        y_pred = y_pred.to(X.device)
        for sublayer in self.output_layer:
            y_by_asp = sublayer(y).unsqueeze(0).requires_grad_(True)
            y_pred = torch.cat((y_pred, y_by_asp), dim=0)
        return y_pred 
