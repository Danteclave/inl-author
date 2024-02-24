#!pip install transformers
# !pip install wandb -qqq
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# from google.colab import drive
# drive.mount('/content/drive')
import json

import numpy as np
import torch.cuda
from sklearn.metrics import precision_recall_fscore_support

import wandb
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm
from transformers import BertTokenizer, AdamW

from DirectoryLoader import DirectoryLoader
from MathUtils import hamming_distance
from MultiAutorshipModel import MultiAuthorshipModel

config = {
    "epochs": 4,
    "lrate": 5e-5,
    "hidden_dropout_prob": 0.2,
    "train_batch_size": 16,
    "eval_batch_size": 16,
    "rstate": 42
}

paths = {
    "test": "data/test",
    "train": "data/train",
    "validation": "data/validation"
}

validationLoader: DirectoryLoader = ...
trainLoader: DirectoryLoader = ...
testLoader: DirectoryLoader = ...

name = ""

if input("Load directories or a csv? type c or d\n>") == "d":
    limit = int(input("Limit the rows? write -1 to not limit\n>"))
    validationLoader: DirectoryLoader = DirectoryLoader(dir_path=paths["validation"]).load(limit=limit).writeCollated()
    trainLoader: DirectoryLoader = DirectoryLoader(dir_path=paths["train"]).load(limit=limit).writeCollated()
    testLoader: DirectoryLoader = DirectoryLoader(dir_path=paths["test"]).load(limit=limit).writeCollated()
    name = str(limit)

else:
    validationLoader: DirectoryLoader = DirectoryLoader(dir_path=paths["validation"]).load_csv()
    trainLoader: DirectoryLoader = DirectoryLoader(dir_path=paths["train"]).load_csv()
    testLoader: DirectoryLoader = DirectoryLoader(dir_path=paths["test"]).load_csv()
    name = "inf"

print("Should a model be loaded, or trained from scratch? Note that training a new model might require a lot of "
      "computational power and can generate a lot of heat and noise on your GPU.")
fpath = input("Enter a path or leave blank to create a new model.\n>")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model: MultiAuthorshipModel = MultiAuthorshipModel(tokenizer=tokenizer).createModel(fpath)

cuda = False
if torch.cuda.is_available():
    model.model.cuda()
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    cuda = True
    torch.cuda.empty_cache()
else:
    device = torch.device()
model.device = device

##############################################################################################
wandb.login(key="5ac2dbe465c3c1200fdb7495f02b4afdd4f9ee40", force=True, relogin=True)
wandb.init(
    project="INL Authorship Detection",
    config=config,
    name=name + f"-ltd {config['epochs']} epochs {config['lrate']} lrate"
)
##############################################################################################

if len(fpath) == 0:
    param_optimizer = list(model.model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.01
        },
        {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config["lrate"])

    dataloader = trainLoader(mode="pairs", tangler="all", tokenizer=tokenizer, batch_size=config["train_batch_size"])

    model.model.train()
    vloss = 0
    vals = 0
    sumprecision, sumrecall, sumf1, sumacc = 0, 0, 0, 0
    ctr = 0
    for _ in range(config["epochs"]):
        for batch in tqdm(dataloader):
            if cuda:
                batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, labels = batch
            optimizer.zero_grad()
            logits = model(input_ids, token_type_ids=None, attention_mask=input_mask).logits
            loss_func = BCEWithLogitsLoss()
            loss = loss_func(logits, labels.type_as(logits))
            loss.backward()
            optimizer.step()

            preds = np.array(logits.argmax(axis=1).cpu())
            labels = np.array(labels.argmax(axis=1).cpu())
            ctr += preds.shape[0]
            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
            accuracy = (preds == labels).mean()
            sumprecision += precision
            sumrecall += recall
            sumf1 += f1
            sumacc += accuracy
            wandb.log(
                {"train_precision": precision, "train_recall": recall, "train_f1": f1, "train_accuracy": accuracy,
                 "train_avg_precision": sumprecision / ctr, "train_avg_recall": sumrecall / ctr,
                 "train_avg_f1": sumf1 / ctr,
                 "train_avg_acc": sumacc / ctr})

            vloss += loss.item()
            vals += 1
    print(f"Train loss: {vloss / vals}")

if True:
    model.model.eval()
    vloss = 0
    vals = 0
    sumprecision, sumrecall, sumf1, sumacc = 0, 0, 0, 0
    ctr = 0
    for step, batch in enumerate(
            validationLoader(mode="pairs", tangler="all", tokenizer=tokenizer, batch_size=config["eval_batch_size"])):
        if cuda:
            batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            input_ids, input_mask, labels = batch

            # temp
            logits = model(input_ids, token_type_ids=None, attention_mask=input_mask).logits
            loss_func = BCEWithLogitsLoss()
            vloss += loss_func(logits, labels.type_as(logits)).item()
            vals += input_ids.size(0)

            preds = np.array(logits.argmax(axis=1).cpu())
            labels = np.array(labels.argmax(axis=1).cpu())
            ctr += preds.shape[0]
            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
            accuracy = (preds == labels).mean()
            sumprecision += precision
            sumrecall += recall
            sumf1 += f1
            sumacc += accuracy
            wandb.log(
                {"eval_precision": precision, "eval_recall": recall, "eval_f1": f1, "eval_accuracy": accuracy,
                 "eval_avg_precision": sumprecision / ctr, "eval_avg_recall": sumrecall / ctr,
                 "eval_avg_f1": sumf1 / ctr,
                 "eval_avg_acc": sumacc / ctr})

    print(f"Validation loss: {vloss / vals}")

"""
Testing part, this uses custom data loading without using data loaders (architecture could be adapted)
"""
testData = testLoader(mode="concat")
bar = tqdm(total=testData.shape[0],
           bar_format="Running test on full paragraphs: " +
                      '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

total_hamming = 0

TP, FP, TN, FN = 0, 0, 0, 0
acc, P, R, f1 = 0, 0, 0, 0

for i, x in testLoader(mode="concat").iterrows():
    outs = model(full="True", data=x["text"])
    outs_nice = [idx + 1 for idx, sublist in enumerate(outs) for item in sublist]
    total_hamming += hamming_distance(outs_nice, json.loads(x["authors"]))
    actual = all(x == 1 for x in json.loads(x["authors"]))
    predicted = all(x == 1 for x in outs_nice)

    if actual == predicted:
        if actual == 1:
            TP += 1
        else:
            TN += 1
    else:
        if actual == 1:
            FN += 1
        else:
            FP += 1
    acc = (TP+TN)/(TP+FP+TN+FN)
    if TP+FP == 0:
        P = 0
    else:
        P = TP/(TP+FP)
    if TP+FN == 0:
        R = 0
    else:
        R = TP/(TP+FN)
    if P+R == 0:
        f1 = 0
    else:
        f1 = 2*P*R/(P+R)
    wandb.log({"full_acc": acc, "full_precision": P, "full_recall": R, "full_f1": f1,
               "full_TP": TP, "full_FP": FP, "full_TN": TN, "full_FN": FN})

    wandb.log({"total_hamming": total_hamming, "average_hamming": total_hamming / (i + 1)})
    bar.update()
bar.close()

print("Matrix for single(T) vs multi(F) predictions:")
print("--------------")
print(f"P\A| A:T  | A:F")
print(f"P:T| {TP:4d} | {FP:4d}")
print(f"P:F| {FN:4d} | {TN:4d}")
print("--------------")
print(f"acc: {acc}, P: {P}, R: {R}, f1: {f1}")


wandb.finish()

path = input("Save model? Leave blank for no, or input a path\n>")
if len(path) != 0:
    model.saveModel(path)
