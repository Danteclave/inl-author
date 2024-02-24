from dataclasses import dataclass
from itertools import combinations

import pandas as pd
import torch
from transformers import BertForNextSentencePrediction, BertTokenizer
from MathUtils import group_blocks
import numpy as np


@dataclass(init=True)
class MultiAuthorshipModel:
    model: BertForNextSentencePrediction = None  # composition over inheritance
    tokenizer: BertTokenizer = ...
    device: torch.device = ...
    """
    full=True works only on a single string not for tensor datasets or any of that nonsense
    """
    def __call__(self, *args, **kwargs):
        if "full" not in kwargs.keys() or kwargs["full"] is False:
            return self.model(*args, **kwargs)

        # union find magic
        """
        I tried making the union find as reasonable as possible
        but the fact that this requires so many calls to unbatched data can cause issues
        even with shifting computation to gpu
        """
        # tokenizer(list(self.df["text"])
        if "data" not in kwargs:
            raise Exception("Data not provided")
        data = kwargs["data"]

        res = group_blocks(data.split("\n"), comparator=self._compareTwo)
        return [[e[1] for e in x] for x in res]

    def _compareTwo(self, left_text, right_text) -> bool:
        tokenized = self.tokenizer(left_text, right_text, truncation=True, padding=True)
        ids = torch.tensor([tokenized['input_ids']]).to(self.device)
        mask = torch.tensor([tokenized["attention_mask"]]).to(self.device)
        out = self.model(ids, token_type_ids=None, attention_mask=mask)
        return bool(torch.argmax(out.logits))

    def createModel(self, fpath=None):
        if fpath is None or len(fpath) == 0:
            self.model = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")
        else:
            self.model = BertForNextSentencePrediction.from_pretrained(fpath, local_files_only=True)
        return self

    def saveModel(self, fpath):
        if fpath is not None:
            self.model.save_pretrained(fpath)
        else:
            raise Exception("Incorrect path")
