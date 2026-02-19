import os
import json
import random
import sqlite3
from datetime import datetime
from flask import Flask, render_template, jsonify, redirect, url_for, request, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user

# -------------------------------
# App Setup
# -------------------------------

app = Flask(__name__)
app.secret_key = "super_secret_key_123"

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

eeg_history = []
abnormal_count = 0

# -------------------------------
# Disable Cache (IMPORTANT)
# -------------------------------
@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# -------------------------------
# Load ML Model
# -------------------------------
with open("brain_model.json", "r") as f:
    model = json.load(f)

threshold = model["threshold"]
model_accuracy = model["accuracy"]

# -------------------------------
# Initialize Database
# -------------------------------
def init_database():
    conn = sqlite3.connect("eeg_database.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS eeg_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            eeg_value INTEGER,
            status TEXT,
            risk REAL,
            timestamp TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)

    conn.commit()
    conn.close()

init_database()

# -------------------------------
# User Class
# -------------------------------
class User(UserMixin):
    def __init__(self, id, username):
        self.id = str(id)
        self.username = username

@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect("eeg_database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, username FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()

    if user:
        return User(user[0], user[1])
    return None

# -------------------------------
# ML Logic
# -------------------------------
def predict_brain_state(eeg_value):
    return "Abnormal Activity Detected" if eeg_value > threshold else "Normal Brain Activity"

def calculate_risk(eeg_value):
    if eeg_value > threshold:
        risk = (eeg_value - threshold) * 3
        return min(100, round(risk, 2))
    return 0

def get_eeg_from_hardware():
    return random.randint(60, 120)

# -------------------------------
# Logging
# -------------------------------
def log_data(eeg_value, status, risk):
    conn = sqlite3.connect("eeg_database.db")
    cursor = conn.cursor()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        INSERT INTO eeg_records (eeg_value, status, risk, timestamp)
        VALUES (?, ?, ?, ?)
    """, (eeg_value, status, risk, timestamp))

    conn.commit()
    conn.close()

# -------------------------------
# Login
# -------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("eeg_database.db")
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, password FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()

        if user and user[2] == password:
            login_user(User(user[0], user[1]))
            return redirect(url_for("admin"))

    return render_template("login.html")

# -------------------------------
# Logout (FIXED)
# -------------------------------
@app.route("/logout")
def logout():
    logout_user()
    session.clear()
    return redirect(url_for("login"))

# -------------------------------
# Public Dashboard (NO login required)
# -------------------------------
@app.route("/")
def home():
    global eeg_history
    global abnormal_count

    eeg_value = get_eeg_from_hardware()
    status = predict_brain_state(eeg_value)
    risk = calculate_risk(eeg_value)

    if "Abnormal" in status:
        abnormal_count += 1

    eeg_history.append(eeg_value)
    if len(eeg_history) > 20:
        eeg_history.pop(0)

    log_data(eeg_value, status, risk)

    return render_template(
        "index.html",
        eeg=eeg_value,
        status=status,
        history=eeg_history,
        accuracy=model_accuracy,
        threshold=threshold,
        risk=risk,
        abnormal_count=abnormal_count
    )

# -------------------------------
# Protected Admin
# -------------------------------
@app.route("/admin")
@login_required
def admin():
    conn = sqlite3.connect("eeg_database.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM eeg_records
        ORDER BY id DESC
        LIMIT 20
    """)

    records = cursor.fetchall()
    conn.close()

    return render_template("admin.html", records=records)

# -------------------------------
# API
# -------------------------------
@app.route("/api/eeg")
@login_required
def api_eeg():
    eeg_value = get_eeg_from_hardware()
    status = predict_brain_state(eeg_value)
    risk = calculate_risk(eeg_value)

    return jsonify({
        "eeg_value": eeg_value,
        "status": status,
        "risk": risk,
        "threshold": threshold
    })

# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
