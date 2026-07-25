import json
import random
import sqlite3
from datetime import datetime
from flask import Flask, render_template, redirect, url_for, request, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user

# ---------------- APP INIT ----------------
app = Flask(__name__)
app.secret_key = "super_secret_key"

# ---------------- LOGIN MANAGER ----------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# ---------------- LOAD MODEL ----------------
with open("cardiac_model.json", "r") as f:
    model = json.load(f)

threshold = model["threshold"]
accuracy = model["accuracy"]

# ---------------- DATABASE INIT ----------------
# ---------------- DATABASE INIT ----------------
def init_db():
    conn = sqlite3.connect("cardiac_database.db")
    cursor = conn.cursor()

    # Users Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)

    # Cardiac Records Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cardiac_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bpm INTEGER,
            status TEXT,
            risk REAL,
            timestamp TEXT
        )
    """)

    # Default Admin Account
    cursor.execute("SELECT * FROM users WHERE username = ?", ("admin",))
    if cursor.fetchone() is None:
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            ("admin", "admin")
        )

    conn.commit()
    conn.close()

# Initialize Database
init_db()

# ---------------- USER CLASS ----------------
class User(UserMixin):
    def __init__(self, id, username):
        self.id = str(id)
        self.username = username

@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect("cardiac_database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, username FROM users WHERE id=?", (user_id,))
    user = cursor.fetchone()
    conn.close()

    if user:
        return User(user[0], user[1])
    return None

# ---------------- AI LOGIC ----------------
def predict_heart_state(bpm):
    if bpm > threshold:
        return "Abnormal Heart Rate"
    return "Normal"

def calculate_risk(bpm):
    if bpm > threshold:
        return min(100, round((bpm - threshold) * 3, 2))
    return 0

def get_bpm():
    return random.randint(65,100)

def log_data(bpm, status, risk):
    conn = sqlite3.connect("cardiac_database.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO cardiac_records (bpm, status, risk, timestamp)
        VALUES (?, ?, ?, ?)
    """, (bpm, status, risk, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    conn.commit()
    conn.close()

# ---------------- ROUTES ----------------

@app.route("/")
@login_required
def home():

    bpm = get_bpm()

    status = predict_heart_state(bpm)

    risk = calculate_risk(bpm)

    log_data(bpm, status, risk)

    return render_template(
        "index.html",
        bpm=bpm,
        status=status,
        risk=risk,
        accuracy=accuracy
    )

@app.route("/admin")
@login_required
def admin():
    conn = sqlite3.connect("cardiac_database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM cardiac_records ORDER BY id DESC LIMIT 20")
    records = cursor.fetchall()
    conn.close()

    return render_template("admin.html", records=records)

# 🔥 ADD THIS (METAVERSE ROUTE)
@app.route("/metaverse")
@login_required
def metaverse():
    return render_template("metaverse.html")

# ---------------- LOGIN ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("cardiac_database.db")
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

        conn = sqlite3.connect("cardiac_database.db")
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

    bpm = get_bpm()

    status = predict_heart_state(bpm)

    risk = calculate_risk(bpm)

    log_data(bpm, status, risk)

    return jsonify({

        "bpm": bpm,

        "status": status,

        "risk": risk

    })
# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)