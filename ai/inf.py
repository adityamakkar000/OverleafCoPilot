import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf, MISSING
from dataclasses import dataclass

@dataclass
class InferenceConfig:
  path: str = MISSING

cs = ConfigStore.instance()
cs.store(name="inference_config", node=InferenceConfig)

def load_model(cfg: InferenceConfig):
  model = AutoModelForCausalLM.from_pretrained(cfg.model_id, device_map=cfg.device)
  merged_model = PeftModel.from_pretrained(model, cfg.path)
  merged_model = merged_model.merge_and_unload()
  tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, add_eos_token=True, padding_side="left")
  return merged_model, tokenizer

def get_completion(query: str, model, tokenizer, device: str) -> str:
  prompt = r"""
  <start_of_turn>user
  Below is an instruction that describes a task. Write a response that appropriately completes the request.
  {query}
  <end_of_turn>\n<start_of_turn>model

  """.format(query=query)

  encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
  model_inputs = encodeds.to(device)
  generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True, pad_token_id=tokenizer.eos_token_id)
  decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
  return decoded

@hydra.main(config_path=None, config_name="inference_config")
def main(cfg: InferenceConfig):
  model, tokenizer = load_model(cfg)
  instruction = "You are a latex autocomplete model. You will be given a sentence from a proof and you need to finish the sentence. Give back the sentence in latex markup. Here is the sentence to complete: "
  prompt = r"""

  """
  query = f"""{instruction} {prompt}"""
  result = get_completion(query=query, model=model, tokenizer=tokenizer, device=cfg.device)
  print(result)

if __name__ == "__main__":
  main()
