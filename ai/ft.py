import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, MISSING
from ds import get_dataset, DatasetConfig
from dataclasses import dataclass
import os
import shutil


@dataclass
class ftConfig:
    model_id: str = MISSING
    seed: int = MISSING
    test_size: float = MISSING
    modules_limit: int = MISSING
    r: int = MISSING
    lora_alpha: int = MISSING
    dataCFG: DatasetConfig = MISSING
    prefix_txt: str = MISSING
    per_device_train_batch_size: int = MISSING
    gradient_accumulation_steps: int = MISSING
    optim: str = MISSING
    warmup_steps: float = MISSING
    max_steps: int = MISSING
    learning_rate: float = MISSING
    logging_steps: int = MISSING
    output: str = MISSING
    name: str = MISSING
    overwrite: bool = MISSING


class ModelTrainer:
    def __init__(self, cfg: ftConfig):
        self.cfg = cfg
        self.device = self._get_device()
        self.tokenizer = None
        self.model = None

    def _get_device(self):
        if torch.backends.mps.is_available():
            return "mps"
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup_output_directory(self):
        if os.path.exists(f"{self.cfg.output}/{self.cfg.name}/config.yaml"):
            if self.cfg.overwrite:
                shutil.rmtree(f"{self.cfg.output}/{self.cfg.name}/")
            else:
                raise FileExistsError(
                    "Output directory exists. Set overwrite to true to overwrite."
                )

        os.makedirs(f"./{self.cfg.output}/{self.cfg.name}/", exist_ok=True)
        with open(f"./{self.cfg.output}/{self.cfg.name}/config.yaml", "w") as file:
            file.write(OmegaConf.to_yaml(self.cfg))

    def prepare_dataset(self):
        def generate_prompt(data_point):
            prefix_text = self.cfg.prefix_txt
            text = r""" <start_of_turn>user {prefix_text} {instruction}  {input} <end_of_turn> <start_of_turn>model {output} <end_of_turn>""".format(
                prefix_text=prefix_text,
                instruction=data_point["instruction"],
                input=data_point["input"],
                output=data_point["output"],
            )

            return text

        ds = get_dataset(self.cfg.dataCFG)
        text_column = [generate_prompt(data_point) for data_point in ds["train"]]
        ds = ds["train"].add_column("prompt", text_column)
        ds = ds.shuffle(seed=self.cfg.seed)
        ds = ds.map(lambda samples: self.tokenizer(samples["prompt"]), batched=True)
        ds = ds.train_test_split(test_size=self.cfg.test_size)
        return ds["train"], ds["test"]

    def setup_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_id, add_eos_token=True, padding_side="left"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_id, device_map=self.device
        )
        self.model.gradient_checkpointing_enable()

        modules = self._find_all_linear_names()
        target_modules = (
            modules
            if len(modules) < self.cfg.modules_limit
            else modules[: self.cfg.modules_limit]
        )

        lora_config = LoraConfig(
            r=self.cfg.r,
            lora_alpha=self.cfg.lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(self.model, lora_config)
        return lora_config

    def _find_all_linear_names(self):
        lora_module_names = set()
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                names = name.split(".")
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        return list(lora_module_names)

    def train(self):
        self.setup_output_directory()
        lora_config = self.setup_model()
        train_data, test_data = self.prepare_dataset()

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_data,
            eval_dataset=test_data,
            dataset_text_field="prompt",
            peft_config=lora_config,
            max_seq_length=250,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=self.cfg.per_device_train_batch_size,
                gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
                warmup_steps=self.cfg.warmup_steps,
                max_steps=self.cfg.max_steps,
                learning_rate=self.cfg.learning_rate,
                logging_steps=self.cfg.logging_steps,
                output_dir=f"{self.cfg.output}/{self.cfg.name}/checkpoints",
                optim=self.cfg.optim,
                save_strategy="epoch",
            ),
            data_collator=transformers.DataCollatorForLanguageModeling(
                self.tokenizer, mlm=False
            ),
        )

        trainer.train()
        self.save_models(trainer)

    def save_models(self, trainer):
        new_model_path = f"{self.cfg.output}/{self.cfg.name}/finetuned_models/"
        trainer.model.save_pretrained(new_model_path)

        merged_model = PeftModel.from_pretrained(self.model, new_model_path)
        merged_model = merged_model.merge_and_unload()

        merged_path = f"{self.cfg.output}/{self.cfg.name}/merged_models/"
        merged_model.save_pretrained(merged_path)
        self.tokenizer.save_pretrained(merged_path)


cs = ConfigStore.instance()
cs.store(name="base", node=ftConfig)


@hydra.main(version_base=None, config_path="./configs")
def main(cfg: ftConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    trainer = ModelTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
