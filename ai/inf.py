import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, MISSING
from dataclasses import dataclass


@dataclass
class InferenceConfig:
    path: str = MISSING
    model_id: str = MISSING
    output: str = MISSING
    name: str = MISSING


class ModelManager:
    def __init__(self):
        self.device = self._get_device()

    @staticmethod
    def _get_device():
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load_model(self, cfg: InferenceConfig):
        model = AutoModelForCausalLM.from_pretrained(cfg.model_id, device_map=self.device)
        merged_model = PeftModel.from_pretrained(
            model, f"{cfg.output}/{cfg.name}/finetuned_models/"
        )
        merged_model = merged_model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_id, add_eos_token=True, padding_side="left"
        )
        return merged_model, tokenizer

    def get_completion(self, query: str, model, tokenizer) -> str:
        prompt_template = """
        <start_of_turn>user
        Below is an instruction that describes a task. Write a response that appropriately completes the request.
        {query}
        <end_of_turn>\n<start_of_turn>model
        """
        prompt = prompt_template.format(query=query)

        encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        model_inputs = encodeds.to(self.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=50,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


def setup_config_store():
    cs = ConfigStore.instance()
    cs.store(name="inference_config", node=InferenceConfig)


@hydra.main(version_base=None, config_path="./configs")
def main(cfg: InferenceConfig):
    cfg = OmegaConf.load(f"{cfg.path}/config.yaml")
    print(OmegaConf.to_yaml(cfg))

    model_manager = ModelManager()
    model, tokenizer = model_manager.load_model(cfg)

    instruction = "You are a latex autocomplete model. You will be given a sentence from a proof and you need to finish the sentence. Give back the sentence in latex markup. Here is the sentence to complete: "
    prompt = r""" We will argue by contradiction. Let us say we have $G_1$ and $G_2$ such that """
    query = f"{instruction} {prompt}"

    result = model_manager.get_completion(query=query, model=model, tokenizer=tokenizer)
    print(result)


if __name__ == "__main__":
    setup_config_store()
    main()
