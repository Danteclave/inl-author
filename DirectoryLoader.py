import json
from dataclasses import dataclass
import glob
from typing import Callable, Union

import torch.utils.data
from pandas import DataFrame
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from typing_extensions import Self

import pandas as pd
import numpy as np
from pathlib import Path
import time
from itertools import combinations


def defaultTransform(*args):
    val = args[0]
    pos = val.rfind("\\")
    pos = max(pos, 0)
    return (val[:pos] + '\\truth-' + val[pos + 1:]).replace(".txt", ".json")


class PrefixTransformer:
    def __init__(self, method=None):
        if method is None:
            self.method = defaultTransform
        else:
            self.method: Callable[..., str] = method

    def __call__(self, *args, **kwargs) -> str:
        return self.method(args[0])


@dataclass(init=True)
class DirectoryLoader:
    combos: int = 0
    df: DataFrame = None
    truth_prefix: PrefixTransformer = PrefixTransformer()
    dir_path: str = ""
    preprocess = True
    logging = 10
    fmt2nd = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'

    def load_csv(self) -> Self:
        self.preprocess = False  # we assume the data is preprocessed since it's a collation
        self.df = pd.read_csv(f"{self.dir_path}/collated.csv")
        self.combos = self.df["text"].str.count("\n").sum()

        print(f"Loaded frame from {self.dir_path} ({self.df.shape[0]} rows)")

        return self

    def load(self, limit=None) -> Self:
        start_time = time.perf_counter()
        files_json = np.array(glob.glob(self.dir_path + "/*.json"))
        files_txt = np.array(glob.glob(self.dir_path + "/*.txt"))

        if 0 in (len(files_json), len(files_txt)):
            raise Exception(f"Directory {self.dir_path} does not contain any"
                            f"{'.json' if len(files_json) == 0 else '.txt'} files.")

        if len(files_json) != len(files_txt):
            print(f"There is a mismatch between the number of json and txt files in {self.dir_path}."
                  f"The loader will attempt to make things work.")

        if limit is not None and limit != -1:
            files_json = files_json[:limit]
            files_txt = files_txt[:limit]

        # internally we keep it as these
        self.df = pd.DataFrame(np.nan, columns=["text", "authors"],
                               index=range(min(len(files_json), len(files_txt), limit)))

        skipped_txt, skipped_json = [0, 0]
        invalid_files = 0
        valid = -1
        bar = tqdm(total=len(files_txt), bar_format=f'Loading {self.dir_path}: ' + self.fmt2nd)
        for x in files_txt:
            x_json: str = self.truth_prefix(x)
            if x_json not in files_json:
                skipped_txt += 1
                bar.update()
                continue

            pt = Path(x)
            pj = Path(x_json)
            invalid_files += (not pj.exists()) + (not pt.exists())
            row = {"text": pt.read_text(encoding="utf8"),
                   "authors": str(json.loads(pj.read_text())["paragraph-authors"])}
            if self.preprocess:
                row["text"] = row["text"].lower().strip().encode("ascii", "ignore").decode()
            h = row["text"].count("\n") + 1
            self.combos += (h * (h - 1)) // 2
            self.df.loc[(valid := valid + 1)] = row
            bar.update()

        bar.close()

        self.df.dropna()

        skipped_json = len(files_json) - self.df.shape[0]

        print(f"Loader for {self.dir_path}; {self.df.shape[0]} rows good in {valid + 1} files; "
              f"skipped {skipped_txt} .txt files, {skipped_json} .json files, "
              f"{invalid_files} invalid files took {(time.perf_counter() - start_time) * 1000:.2f}ms")

        return self

    def getDataFrame(self) -> DataFrame:
        return DataFrame({"text": self.df["text"], "authors": self.df["authors"]})

    def getPairs(self, tangler="all", tokenizer=None, write_tangled=True) -> torch.utils.data.Dataset:
        start_time = time.perf_counter()
        comboframe: DataFrame = ...

        if tangler not in ["all", "fnl", "first", "preloaded"]:
            raise Exception(f"tangler was {tangler}; should be 'all', 'fnl', 'first")
        elif tangler == "fnl":
            raise Exception("not implemented")
        elif tangler == "preloaded":
            comboframe = pd.read_csv(self.dir_path+"/paired.csv")
        elif tangler == "all":
            print("running in 'all' mode: this will make .5n(n-1) rows of each text of n paragraphs and WILL eat your "
                  "ram")
            comboframe = DataFrame(np.nan, columns=["left", "right", "same"], index=range(self.combos))
    
            last = 0

            bar = tqdm(total=self.combos, bar_format=f'Tangling for {self.dir_path}: ' + self.fmt2nd)

            # TODO: generally very slow, consider making this better
            for _, x in self.df.iterrows():
                parags = x["text"].split("\n")
                nums = json.loads(x["authors"])
                for y in combinations(enumerate(parags), 2):
                    comboframe.loc[(last := last + 1)] = \
                        dict(zip(["left", "right", "same"], [*[e[1] for e in y], nums[y[0][0]] == nums[y[1][0]]]))
                if tangler == "first":
                    break
                bar.update(sum(1 for _ in combinations(enumerate(parags), 2)))
            bar.close()

        comboframe = comboframe.dropna()

        if write_tangled:
            comboframe.to_csv(self.dir_path + "/paired.csv")

        encoded = tokenizer(list(comboframe["left"]), list(comboframe["right"]), truncation=True, padding=True)

        out = TensorDataset(torch.tensor(encoded['input_ids']),
                            torch.tensor(encoded['attention_mask']),
                            torch.tensor(pd.get_dummies(comboframe["same"]).values, dtype=torch.long))

        print(f"Loading pairs for {self.dir_path} took {(time.perf_counter() - start_time) * 1000}ms")

        return out

    def __call__(self, mode="concat", tangler="all", tokenizer=None, batch_size=16, shuffle=True, write_tangled=True,
                 *args, **kwargs) -> \
            Union[DataLoader, DataFrame]:
        if mode not in ["concat", "pairs"]:
            raise Exception(f"mode was {mode}; should be 'concat', 'pairs'")
        if mode == "pairs":
            return DataLoader(self.getPairs(tangler, tokenizer, write_tangled), batch_size=batch_size, shuffle=shuffle)
        else:
            return self.getDataFrame()

    def writeCollated(self) -> Self:
        self.df.to_csv(self.dir_path + "/collated.csv")
        return self
