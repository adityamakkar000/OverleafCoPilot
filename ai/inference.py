import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, MISSING
from dataclasses import dataclass
from finetune import ftConfig
import time
import os
from dotenv import load_dotenv


@dataclass
class InferenceConfig:
    path: str = MISSING
    tokens: int = 100

class ModelManager:
    def __init__(self, cfg: ftConfig, tokens:int):
        self.device = self._get_device()
        self.model, self.tokenizer = self.load_model(cfg)
        self.model.eval()
        self.device = self._get_device()
        self.instruction = cfg.dataCFG.instruction
        self.tokens = tokens

    @staticmethod
    def _get_device():
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load_model(self, cfg: InferenceConfig):
        model = AutoModelForCausalLM.from_pretrained(cfg.model_id,
                                                     torch_dtype=torch.bfloat16,
                                                     device_map=self.device)
        merged_model = PeftModel.from_pretrained(
            model, f"{cfg.output}/{cfg.name}/finetuned_models/"
        )
        try:
            merged_model = torch.compile(merged_model)
            print("Model compiled with torch.compile!")
        except Exception as e:
            print(f"torch.compile not used: {e}")

        merged_model = merged_model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_id, add_eos_token=True, padding_side="left"
        )
        return merged_model, tokenizer

    def generate_prompt(self, input: str) -> str:
        message = [{
            "role": "user",
            "content": f"{self.instruction} {input}"
        }]

        return self.tokenizer.apply_chat_template(message, add_generation_prompt=True, return_tensors="pt")

    def __call__(self, input: str) -> str:
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        input_tensor = self.generate_prompt(input)
        outputs = self.model.generate(
            input_tensor.to(self.device),
            max_new_tokens=self.tokens,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
        )

        text = self.tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
        return text

def setup_config_store():
    cs = ConfigStore.instance()
    cs.store(name="inference_config", node=InferenceConfig)

@hydra.main(version_base=None, config_path="./configs", config_name="inference")
def main(cfg: InferenceConfig):
    print(OmegaConf.to_yaml(cfg))
    tokens = cfg.tokens
    cfg = OmegaConf.load(f"{cfg.path}/config.yaml")
    print(OmegaConf.to_yaml(cfg))

    model_manager = ModelManager(cfg, tokens)

    load_dotenv()
    prompt_input = r"""{}""".format(os.getenv("PROMPT"))
    print(prompt_input)

    for _ in range(10):
        start = time.time()
        result = model_manager(
            prompt_input
        )
        print(f"\n{result}")
        print(f"Time taken on step {_}: {time.time() - start}")



if __name__ == "__main__":
    setup_config_store()
    main()
