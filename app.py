import csv
import os
import json
import random
from flask import Flask, render_template, jsonify

app = Flask(__name__)

eeg_history = []

# -------- Load Trained Model --------
with open("brain_model.json", "r") as f:
    model = json.load(f)

threshold = model["threshold"]
model_accuracy = model["accuracy"]

# -------- ML Prediction Layer --------
def predict_brain_state(eeg_value):
    if eeg_value > threshold:
        return "Abnormal Activity Detected"
    else:
        return "Normal Brain Activity"

# -------- Simulated Hardware Layer --------
def get_eeg_from_hardware():
    # For now simulated
    return random.randint(60, 120)

# -------- Data Logging Function --------
def log_data(eeg_value, status):
    file_exists = os.path.isfile("eeg_data.csv")

    with open("eeg_data.csv", mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(["EEG_Value", "Status"])

        writer.writerow([eeg_value, status])

# -------- Main Dashboard Route --------
@app.route("/")
def home():
    global eeg_history

    eeg_value = get_eeg_from_hardware()

    eeg_history.append(eeg_value)
    if len(eeg_history) > 20:
        eeg_history.pop(0)

    status = predict_brain_state(eeg_value)

    log_data(eeg_value, status)

    return render_template(
        "index.html",
        eeg=eeg_value,
        status=status,
        history=eeg_history,
        accuracy=model_accuracy,
        threshold=threshold
    )

# -------- API Route (For Hardware / Mobile / IoT) --------
@app.route("/api/eeg")
def api_eeg():
    eeg_value = get_eeg_from_hardware()
    status = predict_brain_state(eeg_value)

    return jsonify({
        "eeg_value": eeg_value,
        "status": status,
        "threshold": threshold
    })

if __name__ == "__main__":
    app.run(debug=True)
