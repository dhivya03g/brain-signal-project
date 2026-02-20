import random
import json

# -----------------------
# 1️⃣ Generate Training Data
# -----------------------

data = []
labels = []

for _ in range(500):
    eeg = random.randint(60, 120)

    # Simulated ground truth labeling
    if eeg > 100:
        label = 1   # Abnormal
    else:
        label = 0   # Normal

    data.append(eeg)
    labels.append(label)

# -----------------------
# 2️⃣ Calculate Threshold (Midpoint Strategy)
# -----------------------

normal_values = [data[i] for i in range(len(data)) if labels[i] == 0]
abnormal_values = [data[i] for i in range(len(data)) if labels[i] == 1]

avg_normal = sum(normal_values) / len(normal_values)
avg_abnormal = sum(abnormal_values) / len(abnormal_values)

# Better threshold = midpoint between class means
threshold = (avg_normal + avg_abnormal) / 2

# -----------------------
# 3️⃣ Calculate Training Accuracy
# -----------------------

correct = 0

for i in range(len(data)):
    prediction = 1 if data[i] > threshold else 0
    if prediction == labels[i]:
        correct += 1

accuracy = correct / len(data)

# -----------------------
# 4️⃣ Save Model
# -----------------------

model = {
    "threshold": threshold,
    "accuracy": accuracy,
    "avg_normal": avg_normal,
    "avg_abnormal": avg_abnormal
}

with open("brain_model.json", "w") as f:
    json.dump(model, f)

print("Model trained successfully!")
print("Threshold:", threshold)
print("Training Accuracy:", accuracy)
print("Model saved as brain_model.json")
