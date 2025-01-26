from dataclasses import dataclass
from typing import List, Dict
import datasets as d
import csv
from omegaconf import MISSING


@dataclass
class DatasetConfig:
    instruction: str = MISSING
    inputPath: str = MISSING
    outputPath: str = MISSING
    cutoff: int = 7


class DatasetProcessor:
    def __init__(self, config: DatasetConfig):
        self.config = config

    def read_data(self) -> List[str]:
        with open(self.config.inputPath, "r") as file:
            data = file.read()
        return self._preprocess_data(data)

    def _preprocess_data(self, data: str) -> List[str]:
        sentences = data.split(".")
        return [
            x.strip("\n\\n").replace("\n", " ").replace("  ", " ") for x in sentences
        ]

    def create_dataset(self, data: List[str]) -> List[Dict]:
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

        return dataset

    def save_dataset(self, dataset: List[Dict]) -> None:
        with open(
            self.config.outputPath, mode="w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=["instruction", "input", "output"])
            writer.writeheader()
            writer.writerows(dataset)


def process_dataset(config: DatasetConfig) -> None:
    processor = DatasetProcessor(config)
    data = processor.read_data()
    dataset = processor.create_dataset(data)
    processor.save_dataset(dataset)


def get_dataset(config: DatasetConfig):
    process_dataset(config)
    ds = d.load_dataset(config.outputPath[-3:], data_files=config.outputPath)
    return ds


if __name__ == "__main__":
    dataCFG = {
        "instruction": "You are a latex autocomplete model. You will be given a sentence from a proof and you need to finish the sentence. Give back the sentence in latex markup. Here is the sentence",
        "inputPath": "./dataset/latex.txt",
        "outputPath": "./dataset/latex.csv",
        "cutoff": 7,
    }

    cfg = DatasetConfig(
        instruction=dataCFG["instruction"],
        inputPath=dataCFG["inputPath"],
        outputPath=dataCFG["outputPath"],
    )

    process_dataset(cfg)
