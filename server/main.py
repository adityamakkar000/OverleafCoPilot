from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/', methods=['GET'])
def home():
    return "Hello World", 200

@app.route('/post', methods=['GET', 'POST'])
def handle_post():
    if request.method == 'GET':
        return "post page", 200
    if request.method == 'POST':
        data = request.get_json()
        print("Received data:", request.data)  # Debugging
        print("Parsed JSON:", data)  # Debugging
        return jsonify({'data': "generated text"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)