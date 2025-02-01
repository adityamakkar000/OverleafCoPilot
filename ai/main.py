import hydra
from omegaconf import OmegaConf
from flask import Flask, jsonify, request
from flask_cors import CORS
from inference import ModelManager, InferenceConfig, setup_config_store
import regex as re
import time


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
    prompt_input = r"""
        If we add $W_1 + W_2$ we obtain $\begin{pmatrix}
        a + \alpha & -a + \beta \\
        b - \alpha & c + \gamma \\
        \end{pmatrix}$. Thus we can set $u= a + \alpha, v = -a + \beta, t = b - \alpha, z = c + \gamma \in F$ and we obtain $W_1 + W_2$ is off the form $\begin{pmatrix}
            u & v \\ t & z \\
        \end{pmatrix}$ and thus we have 4 separate scalars. Therefore if we have the set $S = \{ \begin{pmatrix}
            1 & 0 \\ 0 & 0 \\
        \end{pmatrix}, \begin{pmatrix}
            0 & 1 \\ 0 & 0 \\
        \end{pmatrix} , \begin{pmatrix}
            0 & 0 \\ 1 & 0 \\
        \end{pmatrix}, \begin{pmatrix}
            0 & 0 \\ 0 & 1 \\
        \end{pmatrix}\}$ and $t_1, t_2, t_3, t_4 \in F$ then as a linear combination we obtain $0 = \begin{pmatrix}
            t_1 & t_2 \\ t_3 & t_4
        \end{pmatrix}$
    """

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
