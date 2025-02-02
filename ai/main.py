import hydra
from omegaconf import OmegaConf
from flask import Flask, jsonify, request
from flask_cors import CORS
from inference import ModelManager, InferenceConfig, setup_config_store
import regex as re
import time
import os
from dotenv import load_dotenv


setup_config_store()
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/post', methods=['POST'])
def handle_post():
    if request.method == 'POST':
        data = request.get_json()
        if model_manager is None:
            return jsonify({'error': 'Model not loaded'}), 500
        if 'text' not in data:
            return jsonify({'error': 'No text field in request'}), 400
        inp = re.sub(r'(\\pagebreak|\\section\{.*?\}|\n|\\\\+)', ' ', data['text']).strip()
        print(f"calling model on: {inp}")
        result = model_manager(inp)
        print(result)
        return jsonify({'data': result}), 200

def warmup(n):
    load_dotenv()
    prompt_input = r"""{}""".format(os.getenv("PROMPT"))
    for _ in range(n):
        start = time.time()
        model_manager(
            prompt_input
        ) # warmup run
        print(f"Time taken: {time.time() - start}")

@hydra.main(version_base=None, config_path="./configs", config_name="inference")
def main(cfg: InferenceConfig):
    print(OmegaConf.to_yaml(cfg))
    tokens = cfg.tokens
    cfg = OmegaConf.load(f"{cfg.path}/config.yaml")
    print(OmegaConf.to_yaml(cfg))

    global model_manager
    model_manager = ModelManager(cfg, tokens)
    warmup(3)


    app.run(host="0.0.0.0", port=3000, debug=True, use_reloader=False)


if __name__ == "__main__":
    setup_config_store()
    main()
