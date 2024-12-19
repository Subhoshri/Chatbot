from flask import Flask, request, jsonify, render_template
from chat_response import response

#Flask app          
app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_input = data.get('message')
    if not user_input:
        return jsonify({"error": "No message provided"}), 400
    response_text = response(user_input)
    return jsonify({"response": response_text})

if __name__ == "__main__":
    app.run(debug=True)
#def home():
 #   return jsonify({"message": "Chatbot API is running! Use POST /predict to chat."})