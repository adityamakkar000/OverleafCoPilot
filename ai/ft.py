import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf, MISSING
from ds import get_dataset, datasetConfig
from dataclasses import dataclass

@dataclass
class ftConfig:
  model_id: str = MISSING
  seed: int = MISSING
  test_size: float = MISSING
  modules_limit: int = MISSING
  r: int = MISSING
  lora_alpha: int = MISSING
  dataCFG: datasetConfig = MISSING
  prefix_txt: str = MISSING
  device: str = MISSING
  per_device_train_batch_size: int = MISSING
  gradient_accumulation_steps: int = MISSING
  optim: str = MISSING
  warmup_steps: float = MISSING
  max_steps: int = MISSING
  learning_rate: float = MISSING
  logging_steps: int = MISSING
  output: str = MISSING
  name: str = MISSING

cs = ConfigStore.instance()
cs.store(name="base", node=ftConfig)

@hydra.main(version_base=None, config_path="./configs")
def main(cfg: ftConfig) -> None:

  print(OmegaConf.to_yaml(cfg))

  model_id = cfg.model_id
  tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True, padding_side='left')
  dataset = get_dataset(cfg.dataCFG)

  # hyperparameters
  seed = cfg.seed
  test_size = cfg.test_size
  modules_limit = cfg.modules_limit
  r = cfg.r
  lora_alpha = cfg.lora_alpha
  output_path = cfg.output
  hyperparameters = {
    "per_device_train_batch_size": cfg.per_device_train_batch_size,
    "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
    "optim": cfg.optim,
    "warmup_steps": cfg.warmup_steps,
    "max_steps": cfg.max_steps,
    "learning_rate": cfg.learning_rate,
    "logging_steps": cfg.logging_steps,
  }
  name = cfg.name
  device = cfg.device


  def generate_prompt(data_point):
      prefix_text = cfg.prefix_txt
      text = text = (
          r""" <start_of_turn>user {prefix_text} {instruction}  {input} <end_of_turn> <start_of_turn>model {output} <end_of_turn>""".format(
              prefix_text=prefix_text,
              instruction=data_point["instruction"],
              input=data_point["input"],
              output=data_point["output"],
          )
      )

      return text


  text_column = [generate_prompt(data_point) for data_point in dataset["train"]]
  dataset = dataset["train"].add_column("prompt", text_column)

  dataset = dataset.shuffle(seed=seed)
  dataset = dataset.map(lambda samples: tokenizer(samples["prompt"]), batched=True)
  dataset = dataset.train_test_split(test_size=test_size)
  train_data = dataset["train"]
  test_data = dataset["test"]

  model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device)
  model.gradient_checkpointing_enable()


  def find_all_linear_names(model):
      cls = torch.nn.Linear
      lora_module_names = set()
      for name, module in model.named_modules():
          if isinstance(module, cls):
              names = name.split(".")
              lora_module_names.add(names[0] if len(names) == 1 else names[-1])
      return list(lora_module_names)


  modules = find_all_linear_names(model)


  lora_config = LoraConfig(
      r=r,
      lora_alpha=lora_alpha,
      target_modules=modules if len(modules) < modules_limit else modules[:modules_limit],
      lora_dropout=0.05,
      bias="none",
      task_type="CAUSAL_LM",
  )

  model = get_peft_model(model, lora_config)
  trainable, total = model.get_nb_trainable_parameters()
  print(
      f"Trainable: {trainable} | total: {total} | Percentage: {trainable/total*100:.4f}%"
  )

  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = "left"


  trainer = SFTTrainer(
      model=model,
      train_dataset=train_data,
      eval_dataset=test_data,
      dataset_text_field="prompt",
      peft_config=lora_config,
      max_seq_length=250,
      args=transformers.TrainingArguments(
          per_device_train_batch_size=hyperparameters["per_device_train_batch_size"],
          gradient_accumulation_steps=hyperparameters["gradient_accumulation_steps"],
          warmup_steps=hyperparameters["warmup_steps"],
          max_steps=hyperparameters["max_steps"],
          learning_rate=hyperparameters["learning_rate"],
          logging_steps=hyperparameters["logging_steps"],
          output_dir=f"./{output_path}/{name}/checkpoints",
          optim=hyperparameters["optim"],
          save_strategy='epoch',
      ),
      data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
  )

  trainer.train()
  new_model = f"./{output_path}/{name}/finetuned_models/"
  trainer.model.save_pretrained(new_model)

  merged_model = PeftModel.from_pretrained(model, new_model)
  merged_model = merged_model.merge_and_unload()

  merged_path = f"./{output_path}/{name}/merged_models/"
  merged_model.save_pretrained(merged_path)
  tokenizer.save_pretrained(merged_path)
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = "right"

if __name__ == "__main__":
  main()