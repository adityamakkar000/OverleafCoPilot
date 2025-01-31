from dataclasses import dataclass
from typing import List, Dict
import datasets as d
import csv
from omegaconf import MISSING
import os
from transformers import AutoTokenizer


@dataclass
class DatasetConfig:
    instruction: str = MISSING
    inputPath: str = MISSING
    outputPath: str = MISSING
    processPath: str = MISSING
    cutoff: int = 7


class DatasetProcessor:
    def __init__(self, config: DatasetConfig):
        self.config = config

    def read_data(self) -> List[str]:
        with open(self.config.inputPath, "r") as file:
            data = file.read()
        self.dataset = self._preprocess_data(data)

    def _preprocess_data(self, data: str) -> List[str]:
        sentences = data.split(".")
        return [
            x.strip("\n\\n").replace("\n", " ").replace("  ", " ") for x in sentences
        ]

    def create_dataset(self, tokenizer=None):
        data = self.dataset
        dataset = []
        length = len(data)
        for i in range(length):
            sentence = data[i]
            if len(sentence) - 1 < self.config.cutoff:
                continue

            indexes = [x for x in range(self.config.cutoff, len(sentence) - 1)]
            input = [sentence[i:_] for _ in indexes]
            output = [sentence[_:] for _ in indexes]
            for a, b in zip(input, output):
                dataset.append(
                    {
                        "instruction": f"{self.config.instruction}",
                        "input": r"{}".format(a),
                        "output": r"{}".format(b),
                    }
                )

        if tokenizer is not None:
            self.tokenize(tokenizer)
        self.save_dataset(dataset)

    def save_dataset(self, dataset: List[Dict]) -> None:
        with open(
            self.config.outputPath, mode="w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=["instruction", "input", "output"])
            writer.writeheader()
            writer.writerows(dataset)

    def generate_prompt(self, data_point, tokenizer):
        message =  [{
            "role": "user",
            "content": f"""{data_point["instruction"]} {data_point["input"]}"""
            },
        {
            "role": "assistant",
            "content": data_point["output"]
            }
            ]

        prompt = tokenizer.apply_chat_template(message, tokenize=False)
        tokenized_prompt = tokenizer(prompt, return_tensors="pt")

        text = {
            'prompt': prompt,
            **tokenized_prompt
        }
        return text

    def tokenize(self, tokenizer):
        ds = d.load_dataset("csv", data_files=self.config.outputPath)["train"]
        ds = ds.map(lambda samples: self.generate_prompt(samples, tokenizer), batched=False)
        print(self.config.processPath)
        ds.save_to_disk(self.config.processPath)

        print("saved to disk")

    def __call__(self, tokenizer):

        self.read_data()
        self.create_dataset(tokenizer)

def get_dataset(config: DatasetConfig, tokenizer=None):

    path = config.processPath if tokenizer is not None else config.outputPath
    if not os.path.exists(path):
        DatasetProcessor(config)(tokenizer)

    if tokenizer is None:
        ds = d.load_dataset("csv", data_files=path)
    else:
        ds = d.load_from_disk(path)
    return ds

# testing 
if __name__ == "__main__":
    dataCFG = {
        "instruction": "You are a latex autocomplete model. You will be given a sentence from a proof and you need to finish the sentence. Give back the sentence in latex markup. Here is the sentence",
        "inputPath": "./dataset/latex.txt",
        "outputPath": "./dataset/latex.csv",
        "processPath": "./dataset/latex/",
        "cutoff": 7,
    }

    tokenizer = AutoTokenizer.from_pretrained(
        "google/gemma-2b-it", add_eos_token=True, padding_side="right"
    )

    cfg = DatasetConfig(
        instruction=dataCFG["instruction"],
        inputPath=dataCFG["inputPath"],
        outputPath=dataCFG["outputPath"],
        processPath=dataCFG["processPath"],
    )

    ds = get_dataset(cfg, tokenizer)
    print(ds)
