import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset
import os
import sys
import time
from datetime import timedelta
from model import MultiTaskClassify


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
ROOT_PATH = os.path.abspath(os.curdir)


def print_progress(index, total, fi="", last=""):
    percent = ("{0:.1f}").format(100 * ((index) / total))
    fill = int(30 * (index / total))
    spec_char = ["\x1b[1;31;40m╺\x1b[0m",
                 "\x1b[1;36;40m━\x1b[0m", "\x1b[1;37;40m━\x1b[0m"]
    bar = spec_char[1]*(fill-1) + spec_char[0] + spec_char[2]*(30-fill)
    if fill == 30:
        bar = spec_char[1]*fill

    percent = " "*(5-len(str(percent))) + str(percent)

    if index == total:
        print(fi + " " + bar + " " + percent + "% " + last)
    else:
        print(fi + " " + bar + " " + percent + "% " + last, end="\r")


class LabelSmoothing(torch.nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        eps = smoothing / num_classes
        self.negative = eps
        self.positive = (1 - smoothing) + eps

    def forward(self, y_pred, y_true):
        y_pred = y_pred.log_softmax(dim=1)
        true_dist = torch.zeros_like(y_pred)
        true_dist.fill_(self.negative)
        true_dist.scatter_(1, y_true.data.unsqueeze(1), self.positive)
        return torch.sum(-true_dist * y_pred, dim=1).mean()


class Trainer:

    def __init__(self, train_loader, valid_loader):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_loader = train_loader
        self._valid_loader = valid_loader
        self.epochs = 8

        self.model = MultiTaskClassify().to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_loader)*8
        )
        self.criterion = nn.CrossEntropyLoss()

        self.train_loss = []
        self.valid_loss = []
        self.train_acc = []
        self.valid_acc = []

    def load_model(self, path):
        if path[0] != "/":
            path = "/" + path
        params = torch.load(ROOT_PATH+path)
        del self.model, self.optimizer, self.scheduler
        self.model = MultiTaskClassify().to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=2e-3)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.train_loader)*20
        )
        self.model.load_state_dict(params["model_state_dict"])
        self.optimizer.load_state_dict(params["optimizer_state_dict"])
        self.scheduler.load_state_dict(params["scheduler_state_dict"])

        self.train_loss, self.valid_loss, self.train_acc, self.valid_acc = params[
            "train_loss"], params["valid_loss"], params["train_acc"], params["valid_acc"]
        print("[INFO] Loaded model successfully")

    def save_model(self, path):
        if path[0] != "/":
            path = "/" + path
        params = {
            "train_loss": self.train_loss,
            "valid_loss": self.valid_loss,
            "train_acc": self.train_acc,
            "valid_acc": self.valid_acc,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
        }
        torch.save(params, ROOT_PATH+path)

    def accuracy_calc(self, y_pred, y_true):
        pred_mask_f1 = torch.where(y_pred == 0, 0, 1).long()
        true_mask_f1 = torch.where(y_true == 0, 0, 1).long()
        final_score = 0
        for col in range(6):
            tp = sum([1 for idx in range(y_pred.size(0)) if pred_mask_f1[idx]
                     [col] == 1 and pred_mask_f1[idx][col] == true_mask_f1[idx][col]])
            if tp == 0:
                continue
            precision_denom = pred_mask_f1[:, col].sum().item()
            recall_denom = true_mask_f1[:, col].sum().item()

            precision = tp/precision_denom
            recall = tp/recall_denom

            f1_score = (2*precision*recall)/(precision+recall)

            count_none_value = (y_pred[:, col] == 0.0).sum().item()
            y_pred[:, col] = torch.where(
                y_pred[:, col] != 0.0, y_pred[:, col], y_true[:, col])
            rss = ((y_true[:, col] - y_pred[:, col])**2).sum().item()
            k = 16*(y_true.size(0) - count_none_value)

            r2_score = 1 - rss/k
            final_score += f1_score*r2_score
        return max(final_score/6, 0)

    def combine_loss(self, y_pred, y_true, training=True):
        loss_sum = torch.Tensor().requires_grad_(training).to(y_pred.device)
        for asp_id in range(6):
            y_pred_asp = y_pred[asp_id]
            loss = self.criterion(
                y_pred_asp, y_true[:, asp_id].long()).view(1,)
            loss_sum = torch.cat((loss_sum, loss), dim=0)
        return torch.sum(loss_sum, dim=0)

    def train_step(self):
        self.model.train()
        loss_his = []
        acc_his = []
        for idx, (X_batch, y_batch) in enumerate(self.train_loader):
            st = time.time()

            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            predict = self.model(X_batch)

            self.optimizer.zero_grad()

            loss = self.combine_loss(predict, y_batch)
            loss.backward()
            loss_his.append(loss.item())

            predict = torch.argmax(predict, dim=2).transpose(0, 1)

            acc = round(self.accuracy_calc(predict, y_batch), 5)
            acc_his.append(acc)

            self.optimizer.step()
            self.scheduler.step()

            calc_time = round(time.time() - st, 1)
            rem_time = calc_time*(len(self.train_loader) - idx - 1)
            eta_time = timedelta(seconds=int(rem_time))
            time_string = f"\x1b[1;31;40m{calc_time}s/step\x1b[0m eta\x1b[1;36;40m {eta_time}\x1b[0m"

            print_progress(idx+1, len(self.train_loader), last=f"Time: {time_string} Loss: {round(loss.item(), 5)} Acc: {acc}",
                           fi=f"Train batch {' '*(len(str(len(self.train_loader)))-len(str(idx+1)))}{idx+1}/{len(self.train_loader)}")

        loss_his = torch.tensor(loss_his)
        acc_his = torch.tensor(acc_his)
        mean_loss = loss_his.mean(dim=0)
        mean_acc = acc_his.mean(dim=0)
        self.train_loss.append(mean_loss.item())
        return mean_loss.item(), mean_acc.item()

    def valid_step(self):
        self.model.eval()
        loss_his = []
        acc_his = []

        with torch.no_grad():
            for idx, (X_batch, y_batch) in enumerate(self.valid_loader):
                st = time.time()

                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                predict = self.model(X_batch)

                loss = self.combine_loss(predict, y_batch, training=False)
                loss_his.append(loss.item())

                predict = nn.Softmax(dim=2)(predict)
                predict = torch.argmax(predict, dim=2).transpose(0, 1)
                acc = round(self.accuracy_calc(predict, y_batch), 5)
                acc_his.append(acc)

                calc_time = round(time.time() - st, 1)
                rem_time = calc_time*(len(self.valid_loader) - idx - 1)
                eta_time = timedelta(seconds=int(rem_time))
                time_string = f"\x1b[1;31;40m{calc_time}s/step\x1b[0m eta\x1b[1;36;40m {eta_time}\x1b[0m"

                print_progress(idx+1, len(self.valid_loader), last=f"Time: {time_string} Loss: {round(loss.item(), 5)} Acc: {acc}",
                               fi=f"Valid batch {' '*(len(str(len(self.valid_loader)))-len(str(idx+1)))}{idx+1}/{len(self.valid_loader)}")

        loss_his = torch.tensor(loss_his)
        acc_his = torch.tensor(acc_his)
        mean_loss = loss_his.mean(dim=0)
        mean_acc = acc_his.mean(dim=0)
        self.valid_loss.append(mean_loss.item())
        return mean_loss.item(), mean_acc.item()


    def fit(self):
        for epoch in range(self.epochs):
            st = time.time()
            print("Epoch:", epoch+1)
            try:
                self.valid_loader = self._valid_loader[epoch]
                train_loss, train_acc = self.train_step()
                valid_loss, valid_acc = self.valid_step()
            except KeyboardInterrupt:
                self.save_model("/dataset/last_model.pt")
                sys.exit()

            calc_time = round(time.time() - st, 1)

            print(f"\t=> Train loss: {round(train_loss, 5)} - Valid loss: {round(valid_loss, 5)} - Train acc: {round(train_acc, 5)} - Valid acc: {round(valid_acc, 5)} - Time: {timedelta(seconds=int(calc_time))}/step\n")
            self.save_model("/dataset/last_model.pt")


def get_loader():
    train_data = torch.load(ROOT_PATH + "/dataset/data.pt")
    train_label = torch.load(ROOT_PATH + "/dataset/label.pt")

    train_loader = DataLoader(
        TensorDataset(train_data, train_label),
        batch_size=4,
        shuffle=True,
        drop_last=False
    )
    valid_loader = DataLoader(
        TensorDataset(train_data[:500], train_label[:500]),
        batch_size=4,
        shuffle=True,
        drop_last=False
    )
    return train_loader, valid_loader


if __name__ == "__main__":
    train_loader, valid_loader = get_loader()
    trainer = Trainer(train_loader, valid_loader)
    trainer.fit()
