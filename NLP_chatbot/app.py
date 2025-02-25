from flask import Flask, request, jsonify
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import json
import requests

# Initialize Flask app
app = Flask(__name__)

# Load trained model and tokenizer
model_path = "./intent_model"  # Ensure this points to the correct directory
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()  # Set model to evaluation mode

# Load intents dataset
with open("intents.json", "r") as f:
    intents = json.load(f)

# Map labels back to intent tags
labels = {intent["tag"]: idx for idx, intent in enumerate(intents["intents"])}
label_map = {idx: tag for tag, idx in labels.items()}

# Function to predict intent
def predict_intent(user_input):
    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_label = torch.argmax(outputs.logits, dim=1).item()
    intent = label_map[predicted_label]
    return intent

# Function to generate a response
def get_response(intent):
    for item in intents["intents"]:
        if item["tag"] == intent:
            return torch.choice(item["responses"])
    return "I'm not sure how to respond to that."

# API route to receive messages
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    intent = predict_intent(user_message)
    response = get_response(intent)

    return jsonify({"intent": intent, "response": response})

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


url = "http://127.0.0.1:5000/chat"
data = {"message": "Hello"}

response = requests.post(url, json=data)
print(response.json())
