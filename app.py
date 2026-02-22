import json
import random
import sqlite3
from datetime import datetime
from flask import Flask, render_template, redirect, url_for, request, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user

app = Flask(__name__)
app.secret_key = "super_secret_key"

# ---------------- LOGIN MANAGER ----------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# ---------------- LOAD MODEL ----------------
with open("brain_model.json", "r") as f:
    model = json.load(f)

threshold = model["threshold"]
accuracy = model["accuracy"]

# ---------------- DATABASE INIT ----------------
def init_db():
    conn = sqlite3.connect("eeg_database.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS eeg_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            eeg_value INTEGER,
            status TEXT,
            risk REAL,
            timestamp TEXT
        )
    """)

    # Default admin
    cursor.execute("SELECT * FROM users WHERE username='admin'")
    if not cursor.fetchone():
        cursor.execute(
            "INSERT INTO users (username,password) VALUES (?,?)",
            ("admin", "admin")
        )

    conn.commit()
    conn.close()

init_db()

# ---------------- USER CLASS ----------------
class User(UserMixin):
    def __init__(self, id, username):
        self.id = str(id)
        self.username = username

@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect("eeg_database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, username FROM users WHERE id=?", (user_id,))
    user = cursor.fetchone()
    conn.close()

    if user:
        return User(user[0], user[1])
    return None

# ---------------- AI LOGIC ----------------
def predict_brain_state(value):
    return "Abnormal Activity Detected" if value > threshold else "Normal Brain Activity"

def calculate_risk(value):
    if value > threshold:
        return min(100, round((value - threshold) * 3, 2))
    return 0

def get_eeg():
    return random.randint(60, 120)

def log_data(value, status, risk):
    conn = sqlite3.connect("eeg_database.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO eeg_records (eeg_value, status, risk, timestamp)
        VALUES (?, ?, ?, ?)
    """, (value, status, risk, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    conn.commit()
    conn.close()

# ---------------- ROUTES ----------------

@app.route("/")
@login_required
def home():
    value = get_eeg()
    status = predict_brain_state(value)
    risk = calculate_risk(value)

    log_data(value, status, risk)

    return render_template(
        "index.html",
        eeg=value,
        status=status,
        risk=risk,
        accuracy=accuracy,
        history=[value]
    )

@app.route("/admin")
@login_required
def admin():
    conn = sqlite3.connect("eeg_database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM eeg_records ORDER BY id DESC LIMIT 20")
    records = cursor.fetchall()
    conn.close()

    return render_template("admin.html", records=records)

# ---------------- LOGIN ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("eeg_database.db")
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, username, password FROM users WHERE username=?",
            (username,)
        )
        user = cursor.fetchone()
        conn.close()

        if user and user[2] == password:
            login_user(User(user[0], user[1]))
            return redirect(url_for("home"))

    return render_template("login.html")

# ---------------- SIGNUP ----------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("eeg_database.db")
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM users WHERE username=?", (username,))
        if cursor.fetchone():
            conn.close()
            return "Username already exists"

        cursor.execute(
            "INSERT INTO users (username,password) VALUES (?,?)",
            (username, password)
        )
        conn.commit()
        conn.close()

        return redirect(url_for("login"))

    return render_template("signup.html")

# ---------------- LOGOUT ----------------
@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

# ---------------- API ----------------
@app.route("/api/health")
@login_required
def api_health():
    value = get_eeg()
    status = predict_brain_state(value)
    risk = calculate_risk(value)

    log_data(value, status, risk)

    return jsonify({
        "eeg": value,
        "status": status,
        "risk": risk
    })

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)