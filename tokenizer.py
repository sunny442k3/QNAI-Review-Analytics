import regex as re
import os
import pandas as pd
import torch
from vncorenlp import VnCoreNLP
from transformers import AutoTokenizer

ROOT_PATH = os.path.abspath(os.curdir)

REGEX_STORES = {
    "EMAIL": re.compile(r"([\w0-9_\.-]+)(@)([\d\w\.-]+)(\.)([\w\.]{2,6})"),
    "URL": re.compile(r"https?:\/\/(?!.*:\/\/)\S+"),
    "PHONE": re.compile(r"(09|01[2|6|8|9])+([0-9]{8})\b"),
    "MENTION": re.compile(r"@.+?:"),
    "NUMBER": re.compile(r"\d+.?\d*"),
    "DATETIME": '\d{1,2}\s?[/-]\s?\d{1,2}\s?[/-]\s?\d{4}',
    "EMOJI1": ":v",
    "EMOJI2": ":D",
    "EMOJI3": ":3",
    "EMOJI4": ":\(",
    "EMOJI5": ":\)"
}
TOKENIZER = AutoTokenizer.from_pretrained(ROOT_PATH + "/dataset/phobert/")
RDRSEGMENT = VnCoreNLP(ROOT_PATH+"/VnCoreNLP/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')


def print_progress(index, total):
    percent = ("{0:.1f}").format(100 * ((index) / total))
    fill = int(50 * (index / total))
    spec_char = ["\u001b[38;5;255m╺\u001b[0m", "\u001b[38;5;198m━\u001b[0m", "\u001b[38;5;234m━\u001b[0m"]
    bar = spec_char[1]*(fill-1) + spec_char[0] + spec_char[2]*(50-fill)
    if fill == 50:
        bar = spec_char[1]*fill
        
    percent = " "*(5-len(str(percent))) + str(percent)
    
    if index == total:
        print(bar + " " + str(percent) + "%")
    else:
        print(bar + " " + str(percent) + "%", end="\r")


class DataProcessing:
    
    def __init__(self):
        self.backup_text = []
        self.stop_words = ['có', 'là', 'rất', 'và', 'mình']
        self.__load_rdrsegment()
        self.__load_tokenizer()
        
        
    def __load_tokenizer(self):
        self.tokenizer = TOKENIZER
        print("[INFO] VinAI/PhoBERT-base load successfully")
        
        
    def __load_rdrsegment(self):
        self.rdrsegment = RDRSEGMENT
        print("[INFO] VnCoreNLP load successfully")
        
        
    def process_data(self, txt):
        try:
            txt = txt.lower()
            txt = re.sub(re.compile(r'<[^>]+>'), ' ', txt)
            txt = re.sub('&.{3,4};', ' ', txt)
            for key, reg in REGEX_STORES.items():
                txt = re.sub(reg, '', txt)
            return txt
        except:
            return ""
        
        
    def convert_data_to_token(self, message, max_length):
        data_seq = self.process_data(message)
        try:
            data_seq = self.rdrsegment.tokenize(data_seq)[0]
        except:
            return []
        mess = " ".join(data_seq)
        mess_encode = self.tokenizer.encode_plus(
            mess,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt',
        )
        return mess_encode["input_ids"]
        
        
    def build_from_csv(self, csv_path):
        if csv_path[0] != "/":
            csv_path = "/" + csv_path
        
        dataframe = pd.read_csv(ROOT_PATH + csv_path, sep=",")
        
        all_mess = dataframe.values[:, 0].tolist()
        label = dataframe.values[:, 1:].tolist()
        
        X_tensor = []
        y_tensor = []
        
        for idx, message in enumerate(all_mess):
            print("Message", idx+1, end="\r")
            seq_tensor = self.convert_data_to_token(message, 256)
            if len(seq_tensor):
                X_tensor.append(seq_tensor.squeeze().tolist())
                y_tensor.append(label[idx])
        X_tensor = torch.tensor(X_tensor)
        y_tensor = torch.tensor(y_tensor)
        shuffle_idx = torch.randperm(X_tensor.size(0))
        X_tensor = X_tensor[shuffle_idx]
        y_tensor = y_tensor[shuffle_idx]
        return X_tensor, y_tensor


if __name__ == "__main__":
    ds = DataProcessing()
    X, y = ds.build_from_csv("./dataset/remake.csv")

    print("Data size:", X.size())
    print("Label size:", y.size())

    torch.save(X, ROOT_PATH + "/dataset/data.pt")
    torch.save(y, ROOT_PATH + "/dataset/label.pt")