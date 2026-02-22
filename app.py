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
    conn.commit()
    conn.close()

init_db()

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
if __name__ == "__main__":
    app.run(debug=True)