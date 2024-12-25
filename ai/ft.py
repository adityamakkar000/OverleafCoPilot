import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer

from ds import get_dataset

model_id = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True)
dataset = get_dataset()

# hyperparameters
seed = 1234
test_size = 0.1
modules_limit = 10
r = 4
lora_alpha = 2


def generate_prompt(data_point):
  prefix_text = 'Below is an instruction that describes a task. Write a response that ' \
              ' completes the request.\n\n'
  text = f"""<start_of_turn>user {prefix_text} {data_point["instruction"]} here is the input {data_point["input"]} <end_of_turn>\n<start_of_turn>model{data_point["output"]} <end_of_turn>"""
  return text

text_column = [generate_prompt(data_point) for data_point in dataset["train"]]
dataset = dataset["train"].add_column("prompt", text_column)

dataset = dataset.shuffle(seed=seed)
dataset = dataset.map(lambda samples: tokenizer(samples["prompt"]), batched=True)
dataset = dataset.train_test_split(test_size=test_size)
train_data = dataset["train"]
test_data = dataset["test"]

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="mps")
model.gradient_checkpointing_enable()

def find_all_linear_names(model):
  cls = torch.nn.Linear
  lora_module_names = set()
  for name, module in model.named_modules():
    if isinstance(module, cls):
      names = name.split('.')
      lora_module_names.add(names[0] if len(names) == 1 else names[-1])
  return list(lora_module_names)

modules = find_all_linear_names(model)


lora_config = LoraConfig(
    r=r,
    lora_alpha=lora_alpha,
    target_modules=modules if len(modules) < modules_limit else modules[:modules_limit],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
trainable, total = model.get_nb_trainable_parameters()
print(f"Trainable: {trainable} | total: {total} | Percentage: {trainable/total*100:.4f}%")

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side='right'

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=test_data,
    dataset_text_field="prompt",
    peft_config=lora_config,
    max_seq_length=2500,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        warmup_steps=0.03,
        max_steps=100,
        learning_rate=2e-4,
        logging_steps=1,
        output_dir="./outputs/checkpoints",
        optim="adamw_torch",
        save_strategy="epoch",
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()
new_model = "./outputs/finetuned_models/gemma-latex-bot"
trainer.model.save_pretrained(new_model)

merged_model= PeftModel.from_pretrained(model, new_model)
merged_model= merged_model.merge_and_unload()

merged_path = f"./outputs/merged_models/{new_model.split('/')[-1]}"
merged_model.save_pretrained(merged_path)
tokenizer.save_pretrained(merged_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
