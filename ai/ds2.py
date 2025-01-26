import datasets as d
import random as r
import csv
import hydra
from omegaconf import DictConfig, OmegaConf, MISSING
from dataclasses import dataclass


@dataclass
class datasetConfig:
    instruction: str = MISSING
    inputPath: str = MISSING
    outputPath: str = MISSING
    cutoff: int = MISSING
    lb: float = MISSING
    up: float = MISSING


def main(dataCFG: datasetConfig):
    instruction = r"""{instruction}""".format(instruction=dataCFG.instruction)
    path = dataCFG.inputPath
    with open(path, "r") as file:
        data = file.read()

    data = data.split(".")
    data = [x.strip("\n\\n").replace("\n", " ").replace("  ", " ") for x in data]

    dataset = []
    cutoff = dataCFG.cutoff
    lb = dataCFG.lb
    up = dataCFG.up

    for i in range(len(data)):
        length = len(data[i])
        if length < cutoff:
            continue

        index = r.randint(int(lb * length), int(up * length))
        dataset.append(
            {
                "instruction": instruction,
                "input": r"{}".format(data[i][:index]),
                "output": r"{}".format(data[i][index:]),
            }
        )

    with open(dataCFG.outputPath, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["instruction", "input", "output"])
        writer.writeheader()
        writer.writerows(dataset)


def get_dataset(dataCFG: datasetConfig):
    main(dataCFG)
    dataset = d.load_dataset(dataCFG.outputPath[-3:], data_files=dataCFG.outputPath)
    return dataset


if __name__ == "__main__":
    main()
