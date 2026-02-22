<<<<<<< HEAD
from flask import Flask, render_template, request, redirect, session, jsonify, flash
import sqlite3, random

app = Flask(__name__)
app.secret_key = "brain_secret"

# ---------------- DATABASE ----------------
def init_db():
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS users(username TEXT, password TEXT)")
    cur.execute("INSERT OR IGNORE INTO users VALUES('admin','admin')")
    conn.commit()
    conn.close()

    conn = sqlite3.connect("eeg_database.db")
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS eeg_data(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        value INTEGER,
        status TEXT,
        risk INTEGER,
        time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
=======
import json
import random
import sqlite3
from datetime import datetime
from flask import Flask, render_template, redirect, url_for, request, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user

app = Flask(__name__)
app.secret_key = "super_secret_key"

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# ------------------------
# Load Model
# ------------------------
with open("brain_model.json", "r") as f:
    model = json.load(f)

threshold = model["threshold"]
accuracy = model["accuracy"]

# ------------------------
# Database Init
# ------------------------
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

>>>>>>> 99c53c573f8131cebfe1f7b9575dc92953571e3f
    conn.commit()
    conn.close()

init_db()

<<<<<<< HEAD
# ---------------- LOGIN ----------------
@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        u = request.form['username']
        p = request.form['password']

        conn = sqlite3.connect("users.db")
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username=? AND password=?", (u,p))
        user = cur.fetchone()
        conn.close()

        if user:
            session['user'] = u
            return redirect('/')
        else:
            flash("Invalid username or password")
            return redirect('/login')

    return render_template("login.html")

# ---------------- DASHBOARD ----------------
@app.route('/')
def dashboard():
    if 'user' not in session:
        return redirect('/login')
    return render_template("dashboard.html")

# ---------------- ADMIN ----------------
@app.route('/admin')
def admin():
    if 'user' not in session:
        return redirect('/login')

    conn = sqlite3.connect("eeg_database.db")
    cur = conn.cursor()
    cur.execute("SELECT * FROM eeg_data ORDER BY id DESC LIMIT 20")
    data = cur.fetchall()
    conn.close()

    return render_template("admin.html", data=data)

# ---------------- LOGOUT ----------------
@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

# ---------------- SENSOR VALUE ----------------
def get_value():
    return random.randint(60,150)

# ---------------- API FOR DASHBOARD ----------------
@app.route('/api/health')
def api_health():
    val = get_value()
    status = "Normal" if val < 120 else "Abnormal"
    risk = 0 if status=="Normal" else 85

    # save history
    conn = sqlite3.connect("eeg_database.db")
    cur = conn.cursor()
    cur.execute("INSERT INTO eeg_data(value,status,risk) VALUES(?,?,?)",(val,status,risk))
    conn.commit()
    conn.close()

    return jsonify({
        "value": val,
        "status": status,
        "risk": risk
    })

# ---------------- RUN ----------------
=======
# ------------------------
# User Class
# ------------------------
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

# ------------------------
# Helper Functions
# ------------------------
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

# ------------------------
# Routes
# ------------------------

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

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("eeg_database.db")
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, password FROM users WHERE username=?", (username,))
        user = cursor.fetchone()
        conn.close()

        if user and user[2] == password:
            login_user(User(user[0], user[1]))
            return redirect(url_for("home"))  # 🔥 Main dashboard

    return render_template("login.html")

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
            return "Username already exists!"

        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        conn.close()

        return redirect(url_for("login"))

    return render_template("signup.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

>>>>>>> 99c53c573f8131cebfe1f7b9575dc92953571e3f
if __name__ == "__main__":
    app.run(debug=True)