from tokenizer import DataProcessing
from model import MultiTaskClassify
import torch
import os 

ROOT_PATH = os.path.abspath(os.curdir)

params = torch.load(ROOT_PATH + "/dataset/model_multitask.pt")
device = torch.device("cpu")
ds = DataProcessing()
model = MultiTaskClassify().to(device)
model.load_state_dict(params["model_state_dict"])


def predict_model(message):
    tokens = ds.convert_data_to_token(message, 256)
    tokens = tokens.to(device)
    model.eval()
    predict = model(tokens)
    predict = torch.argmax(predict, dim=2).transpose(0, 1).squeeze()
    return predict.tolist()


if __name__ == "__main__":
    mess = "Chợ nhỏ, có bán hải sản nhưng không nhiều, giá ok, có điều bán cua cột dây to quá, toàn cua lực sĩ =))"
    print(predict_model(mess))