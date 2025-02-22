from flask import Flask, request, render_template, redirect, url_for, session, flash, send_from_directory
import sqlite3
import os
import re
from werkzeug.security import generate_password_hash, check_password_hash
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from datetime import datetime, timedelta
import shutil
from flask_babel import Babel, gettext, refresh
import pyotp
import requests
import subprocess
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'super-secret-key'  # Replace with a stronger key in production
UPLOAD_FOLDER = 'static/uploads'
AUDIO_FOLDER = 'static/audio'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['AUDIO_FOLDER'] = AUDIO_FOLDER
app.config['OTP_SECRET'] = pyotp.random_base32()
KANNEL_URL = "http://localhost:13013/cgi-bin/sendsms"
KANNEL_USER = "admin"
KANNEL_PASS = "foobar"

# Babel setup
babel = Babel(app)

@babel.localeselector
def get_locale():
    return session.get('lang', 'en')

# Load Mistral model with error handling
# Replace the model loading block:
try:
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"  # Lighter 7B model
    from huggingface_hub import login
    login(token="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")  # Your token
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Failed to load Mistral model: {e}")
    model = None
    tokenizer = None
    embedder = None

# Database Initialization
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    tables = [
        '''CREATE TABLE IF NOT EXISTS teachers (
            id INTEGER PRIMARY KEY, name TEXT, email TEXT UNIQUE, password TEXT,
            subject TEXT, phone TEXT)''',
        '''CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY, name TEXT, roll_no TEXT UNIQUE, class TEXT,
            phone TEXT UNIQUE, profile_image TEXT)''',
        '''CREATE TABLE IF NOT EXISTS academics (
            student_id INTEGER, teacher_id INTEGER, subject TEXT, attendance INTEGER,
            marks INTEGER, remarks TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(student_id) REFERENCES students(id),
            FOREIGN KEY(teacher_id) REFERENCES teachers(id))''',
        '''CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY, teacher_id INTEGER, student_id INTEGER,
            message TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(teacher_id) REFERENCES teachers(id),
            FOREIGN KEY(student_id) REFERENCES students(id))''',
        '''CREATE TABLE IF NOT EXISTS replies (
            id INTEGER PRIMARY KEY, student_id INTEGER, teacher_id INTEGER,
            message TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(student_id) REFERENCES students(id),
            FOREIGN KEY(teacher_id) REFERENCES teachers(id))''',
        '''CREATE TABLE IF NOT EXISTS hod_logs (
            id INTEGER PRIMARY KEY, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            action TEXT, details TEXT)''',
        '''CREATE TABLE IF NOT EXISTS chat_logs (
            student_id INTEGER, message TEXT, response TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(student_id) REFERENCES students(id))''',
        '''CREATE TABLE IF NOT EXISTS resources (
            id INTEGER PRIMARY KEY, teacher_id INTEGER, student_id INTEGER,
            type TEXT, content TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(teacher_id) REFERENCES teachers(id),
            FOREIGN KEY(student_id) REFERENCES students(id))'''
    ]
    for table in tables:
        c.execute(table)
    conn.commit()
    conn.close()

# HOD Terminal (Simplified)
def hod_terminal():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    print("HOD: Initialize 5 teachers first.")
    while c.execute("SELECT COUNT(*) FROM teachers").fetchone()[0] != 5:
        print(f"Current teachers: {c.execute('SELECT COUNT(*) FROM teachers').fetchone()[0]}")
        print("Enter details for 5 teachers (reset if incomplete):")
        c.execute("DELETE FROM teachers")  # Reset if not 5
        for i in range(5):
            print(f"\nTeacher {i+1}/5")
            name = input("Name: ").strip()
            email = input("Email: ").strip()
            if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                print("Invalid email! Starting over.")
                break
            password = input("Password: ").strip()
            subject = input("Subject: ").strip()
            phone = input("Phone (10 digits): ").strip()
            if not re.match(r"^\d{10}$", phone):
                print("Invalid phone! Starting over.")
                break
            if c.execute("SELECT COUNT(*) FROM teachers WHERE email=?", (email,)).fetchone()[0] > 0:
                print("Email exists! Starting over.")
                break
            if c.execute("SELECT COUNT(*) FROM teachers WHERE phone=?", (phone,)).fetchone()[0] > 0:
                print("Phone exists! Starting over.")
                break
            pw_hash = generate_password_hash(password)
            c.execute("INSERT INTO teachers (name, email, password, subject, phone) VALUES (?, ?, ?, ?, ?)",
                      (name, email, pw_hash, subject, phone))
            c.execute("INSERT INTO hod_logs (action, details) VALUES (?, ?)",
                      ("Added Teacher", f"{name} ({email}, {subject})"))
            conn.commit()
        else:
            print("5 teachers initialized!")
            break

    while True:
        print("\nHOD Menu: 1) Add Student 2) Edit Student 3) Backup DB 4) Exit")
        choice = input("Choice: ").strip()
        if choice == '1':
            name = input("Name: ").strip()
            roll_no = input("Roll No: ").strip()
            class_name = input("Class: ").strip()
            phone = input("Phone (10 digits): ").strip()
            if not re.match(r"^\d{10}$", phone) or c.execute("SELECT COUNT(*) FROM students WHERE roll_no=?", (roll_no,)).fetchone()[0] > 0 or c.execute("SELECT COUNT(*) FROM students WHERE phone=?", (phone,)).fetchone()[0] > 0:
                print("Invalid input or duplicate roll_no/phone!")
                continue
            c.execute("INSERT INTO students (name, roll_no, class, phone, profile_image) VALUES (?, ?, ?, ?, ?)",
                      (name, roll_no, class_name, phone, "static/uploads/default.jpg"))
            c.execute("INSERT INTO hod_logs (action, details) VALUES (?, ?)", ("Added Student", f"{name} ({roll_no})"))
            conn.commit()
            print(f"Student {name} ({roll_no}) added!")
        elif choice == '2':
            roll_no = input("Roll No to Edit: ").strip()
            c.execute("SELECT * FROM students WHERE roll_no=?", (roll_no,))
            student = c.fetchone()
            if not student:
                print("Student not found!")
                continue
            name = input(f"Name ({student[1]}): ").strip() or student[1]
            new_roll_no = input(f"Roll No ({student[2]}): ").strip() or student[2]
            class_name = input(f"Class ({student[3]}): ").strip() or student[3]
            phone = input(f"Phone ({student[4]}): ").strip() or student[4]
            if not re.match(r"^\d{10}$", phone) or (new_roll_no != roll_no and c.execute("SELECT COUNT(*) FROM students WHERE roll_no=?", (new_roll_no,)).fetchone()[0] > 0) or (phone != student[4] and c.execute("SELECT COUNT(*) FROM students WHERE phone=?", (phone,)).fetchone()[0] > 0):
                print("Invalid input or duplicate roll_no/phone!")
                continue
            c.execute("UPDATE students SET name=?, roll_no=?, class=?, phone=? WHERE id=?", (name, new_roll_no, class_name, phone, student[0]))
            c.execute("INSERT INTO hod_logs (action, details) VALUES (?, ?)", ("Edited Student", f"{name} ({new_roll_no})"))
            conn.commit()
            print(f"Student {roll_no} updated to {new_roll_no}!")
        elif choice == '3':
            backup_file = f"database_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            shutil.copy('database.db', backup_file)
            c.execute("INSERT INTO hod_logs (action, details) VALUES (?, ?)", ("Backup Database", backup_file))
            conn.commit()
            print(f"Database backed up to {backup_file}")
        elif choice == '4':
            break
        else:
            print("Invalid choice!")
    conn.close()

# Chatbot with TTS
def get_chatbot_response(student_id, user_input):
    if not model or not tokenizer or not embedder:
        return "Chatbot unavailable—model failed to load.", None
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT s.name, s.class, a.subject, a.attendance, a.marks, a.remarks, t.name, n.message, r.content "
              "FROM students s LEFT JOIN academics a ON s.id=a.student_id "
              "LEFT JOIN teachers t ON a.teacher_id=t.id LEFT JOIN notifications n ON s.id=n.student_id "
              "LEFT JOIN resources r ON s.id=r.student_id WHERE s.id=?", (student_id,))
    student_data = c.fetchall()
    c.execute("SELECT message, response FROM chat_logs WHERE student_id=? ORDER BY timestamp DESC LIMIT 3", (student_id,))
    recent_chats = c.fetchall()
    conn.close()

    context = f"Student: {student_data[0][0]}, Class: {student_data[0][1]}\n" + \
              "".join(f"Subject: {row[2]}, Attendance: {row[3]}%, Marks: {row[4]}, Remarks: {row[5]}, Teacher: {row[6]}, Notification: {row[7]}, Resource: {row[8]}\n" for row in student_data) + \
              ("Recent Chat History:\n" + "".join(f"Q: {msg} | A: {resp}\n" for msg, resp in recent_chats) if recent_chats else "")
    
    context_parts = context.split('\n')
    embeddings = embedder.encode([user_input] + context_parts, convert_to_tensor=True)
    similarities = torch.cosine_similarity(embeddings[0:1], embeddings[1:], dim=-1)
    top_indices = similarities.topk(5).indices
    relevant_context = "\n".join(context_parts[i+1] for i in top_indices if similarities[i] > 0.5)

    user_input = user_input or (f"Tell me about my latest {max(student_data, key=lambda x: x[7] or '', default=['']*9)[2]} update." if student_data and max(student_data, key=lambda x: x[7] or '', default=['']*9)[4] else "Hello")
    prompt = f"[INST] You are a chatbot for student {student_data[0][0]}. Use this data only: {relevant_context or context}\nUser: {user_input} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("[/INST]")[-1].strip()
    
    if "not found" in response.lower() or "no data" in response.lower():
        response = "Sorry, that information hasn’t been updated yet."

    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("INSERT INTO chat_logs (student_id, message, response) VALUES (?, ?, ?)", (student_id, user_input, response))
    conn.commit()
    conn.close()

    lang = session.get('lang', 'en')
    lang_map = {'en': 'english', 'hi': 'hindi', 'kn': 'kannada'}
    audio_file = f"response_{student_id}_{int(datetime.now().timestamp())}.wav"
    audio_path = os.path.join(app.config['AUDIO_FOLDER'], audio_file)
    festival_lang = lang_map.get(lang, 'english')
    try:
        subprocess.run(['festival', '--language', festival_lang, '-b', f'(tts_text "{response}" "file" "{audio_path}")'], check=True, timeout=10)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"TTS failed: {e}")
        audio_file = None

    return response, audio_file

# Teacher Routes
@app.route('/teacher', methods=['GET', 'POST'])
def teacher_login():
    if request.method == 'POST':
        email, password = request.form.get('email'), request.form.get('password')
        if not email or not password:
            flash("Email and password required!")
            return render_template('teacher_login.html')
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("SELECT * FROM teachers WHERE email=?", (email,))
        teacher = c.fetchone()
        conn.close()
        if teacher and check_password_hash(teacher[3], password):
            session['teacher_id'] = teacher[0]
            return redirect(url_for('teacher_dashboard'))
        flash("Invalid credentials!")
    return render_template('teacher_login.html')

@app.route('/teacher/logout')
def teacher_logout():
    session.pop('teacher_id', None)
    return redirect(url_for('teacher_login'))

@app.route('/teacher/dashboard', methods=['GET', 'POST'])
def teacher_dashboard():
    if 'teacher_id' not in session:
        return redirect(url_for('teacher_login'))
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT * FROM teachers WHERE id=?", (session['teacher_id'],))
    teacher = c.fetchone()
    if not teacher:
        session.pop('teacher_id', None)
        return redirect(url_for('teacher_login'))
    c.execute("SELECT s.* FROM students s LEFT JOIN academics a ON s.id=a.student_id WHERE a.subject=? OR a.subject IS NULL", (teacher[4],))
    students = c.fetchall()
    
    if request.method == 'POST' and 'bulk_submit' in request.form:
        for student_id in request.form.getlist('student_ids'):
            try:
                attendance = int(request.form.get(f'attendance_{student_id}', '0'))
                marks = int(request.form.get(f'marks_{student_id}', '0'))
                remarks = request.form.get(f'remarks_{student_id}', '')
                if not (0 <= attendance <= 100 and 0 <= marks <= 100):
                    flash(f"Invalid input for student {student_id}—values must be 0-100!")
                    continue
                c.execute("INSERT OR REPLACE INTO academics (student_id, teacher_id, subject, attendance, marks, remarks) VALUES (?, ?, ?, ?, ?, ?)", 
                          (student_id, teacher[0], teacher[4], attendance, marks, remarks))
            except ValueError:
                flash(f"Invalid numeric input for student {student_id}!")
        conn.commit()
    
    conn.close()
    templates = ["Excellent work!", "Needs improvement", "Good effort"]
    return render_template('teacher_dashboard.html', teacher=teacher, students=students, templates=templates)

@app.route('/teacher/profile')
def teacher_profile():
    if 'teacher_id' not in session:
        return redirect(url_for('teacher_login'))
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT * FROM teachers WHERE id=?", (session['teacher_id'],))
    teacher = c.fetchone()
    if not teacher:
        session.pop('teacher_id', None)
        return redirect(url_for('teacher_login'))
    c.execute("SELECT s.name, s.roll_no, a.attendance, a.marks, a.remarks FROM academics a JOIN students s ON a.student_id=s.id WHERE a.teacher_id=?", (teacher[0],))
    academics = c.fetchall()
    c.execute("SELECT s.name, s.roll_no, n.message FROM notifications n JOIN students s ON n.student_id=s.id WHERE n.teacher_id=?", (teacher[0],))
    notifications = c.fetchall()
    conn.close()
    return render_template('teacher_profile.html', teacher=teacher, academics=academics, notifications=notifications)

@app.route('/teacher/student/<roll_no>', methods=['GET', 'POST'])
def student_profile(roll_no):
    if 'teacher_id' not in session:
        return redirect(url_for('teacher_login'))
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT * FROM students WHERE roll_no=?", (roll_no,))
    student = c.fetchone()
    if not student:
        flash("Student not found!")
        return redirect(url_for('teacher_dashboard'))
    c.execute("SELECT subject FROM teachers WHERE id=?", (session['teacher_id'],))
    teacher_subject = c.fetchone()[0]
    if request.method == 'POST':
        if 'academic_submit' in request.form:
            try:
                attendance = int(request.form.get('attendance', '0'))
                marks = int(request.form.get('marks', '0'))
                remarks = request.form.get('remarks', '')
                if not (0 <= attendance <= 100 and 0 <= marks <= 100):
                    flash("Attendance and marks must be 0-100!")
                else:
                    c.execute("INSERT OR REPLACE INTO academics (student_id, teacher_id, subject, attendance, marks, remarks) VALUES (?, ?, ?, ?, ?, ?)", 
                              (student[0], session['teacher_id'], teacher_subject, attendance, marks, remarks))
            except ValueError:
                flash("Invalid numeric input!")
        elif 'message_submit' in request.form:
            message = request.form.get('message', '').strip()
            if message:
                c.execute("INSERT INTO notifications (teacher_id, student_id, message) VALUES (?, ?, ?)", 
                          (session['teacher_id'], student[0], message))
            else:
                flash("Message cannot be empty!")
        elif 'resource_submit' in request.form:
            resource_type = request.form.get('resource_type')
            content = request.form.get('note_content', '').strip() if resource_type == 'note' else None
            if resource_type == 'video':
                file = request.files.get('video_file')
                if file and file.filename.endswith(('.mp4', '.avi', '.mkv')):
                    content = secure_filename(file.filename)
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'], content))
                else:
                    flash("Invalid video file!")
            if content:
                c.execute("INSERT INTO resources (teacher_id, student_id, type, content) VALUES (?, ?, ?, ?)", 
                          (session['teacher_id'], student[0], resource_type, content))
            else:
                flash("Resource content missing!")
        conn.commit()
    c.execute("SELECT a.*, t.name FROM academics a JOIN teachers t ON a.teacher_id=t.id WHERE student_id=?", (student[0],))
    academics = c.fetchall()
    c.execute("SELECT n.*, t.name FROM notifications n JOIN teachers t ON n.teacher_id=t.id WHERE student_id=?", (student[0],))
    notifications = c.fetchall()
    c.execute("SELECT r.message, r.timestamp FROM replies r WHERE student_id=? AND teacher_id=?", (student[0], session['teacher_id']))
    replies = c.fetchall()
    c.execute("SELECT type, content, timestamp FROM resources WHERE student_id=? AND teacher_id=?", (student[0], session['teacher_id']))
    resources = c.fetchall()
    conn.close()
    templates = ["Excellent work!", "Needs improvement", "Good effort"]
    return render_template('student_profile.html', student=student, academics=academics, notifications=notifications, 
                          replies=replies, resources=resources, teacher_subject=teacher_subject, templates=templates)

# Student Routes
@app.route('/', methods=['GET', 'POST'])
def student_login():
    if request.method == 'POST':
        if 'phone' in request.form:
            phone = request.form.get('phone', '').strip()
            if not re.match(r"^\d{10}$", phone):
                flash("Invalid phone number—must be 10 digits!")
                return render_template('student_login.html', step='phone')
            conn = sqlite3.connect('database.db')
            c = conn.cursor()
            c.execute("SELECT * FROM students WHERE phone=?", (phone,))
            student = c.fetchone()
            conn.close()
            if student:
                totp = pyotp.TOTP(app.config['OTP_SECRET'], interval=60)
                otp = totp.now()
                params = {
                    'username': KANNEL_USER,
                    'password': KANNEL_PASS,
                    'to': f"+91{phone}",  # Adjust country code
                    'text': f"Your OTP is {otp}. Valid for 60 seconds."
                }
                try:
                    response = requests.get(KANNEL_URL, params=params, timeout=5)
                    if response.status_code == 202:
                        session['otp_phone'] = phone
                        session['otp_time'] = datetime.now().isoformat()
                        session['otp'] = otp
                        print(f"OTP for {phone}: {otp}")
                        return render_template('student_login.html', step='otp', phone=phone)
                    flash("Failed to send OTP—SMS service issue.")
                except requests.RequestException:
                    flash("Failed to send OTP—check SMS gateway.")
            else:
                flash("Phone number not found!")
        elif 'otp' in request.form:
            phone = session.get('otp_phone')
            entered_otp = request.form.get('otp', '').strip()
            if not phone or not entered_otp:
                flash("Session expired or invalid OTP input!")
                return render_template('student_login.html', step='phone')
            otp_time = datetime.fromisoformat(session.get('otp_time', '2000-01-01'))
            totp = pyotp.TOTP(app.config['OTP_SECRET'], interval=60)
            if (datetime.now() - otp_time).seconds > 60:
                flash("OTP expired!")
                session.clear()
            elif totp.verify(entered_otp, valid_window=0):
                conn = sqlite3.connect('database.db')
                c = conn.cursor()
                c.execute("SELECT id FROM students WHERE phone=?", (phone,))
                student_id = c.fetchone()
                conn.close()
                if student_id:
                    session['student_id'] = student_id[0]
                    session.pop('otp_phone', None)
                    session.pop('otp_time', None)
                    session.pop('otp', None)
                    return redirect(url_for('student_home'))
                flash("Student not found after OTP!")
            else:
                flash("Invalid OTP!")
    return render_template('student_login.html', step='phone')

@app.route('/student/logout')
def student_logout():
    session.pop('student_id', None)
    return redirect(url_for('student_login'))

@app.route('/student/home', methods=['GET', 'POST'])
def student_home():
    if 'student_id' not in session:
        return redirect(url_for('student_login'))
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT * FROM students WHERE id=?", (session['student_id'],))
    student = c.fetchone()
    if not student:
        session.pop('student_id', None)
        return redirect(url_for('student_login'))
    
    if request.method == 'POST' and 'image_submit' in request.form:
        file = request.files.get('profile_image')
        if file and file.filename.endswith(('.jpg', '.jpeg', '.png')):
            filename = secure_filename(f"{student[2]}.jpg")
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            c.execute("UPDATE students SET profile_image=? WHERE id=?", 
                      (os.path.join(app.config['UPLOAD_FOLDER'], filename), student[0]))
            conn.commit()
            flash("Profile image updated!")
        else:
            flash("Invalid image file!")
    
    c.execute("SELECT a.*, t.name FROM academics a JOIN teachers t ON a.teacher_id=t.id WHERE student_id=?", (student[0],))
    academics = c.fetchall()
    c.execute("SELECT n.*, t.name FROM notifications n JOIN teachers t ON n.teacher_id=t.id WHERE student_id=? ORDER BY timestamp DESC", (student[0],))
    notifications = c.fetchall()
    
    if request.method == 'POST' and 'reply_submit' in request.form:
        notification_id = request.form.get('notification_id')
        reply_message = request.form.get('reply_message', '').strip()
        if notification_id and reply_message:
            c.execute("SELECT teacher_id FROM notifications WHERE id=?", (notification_id,))
            teacher_id = c.fetchone()
            if teacher_id:
                c.execute("INSERT INTO replies (student_id, teacher_id, message) VALUES (?, ?, ?)", 
                          (student[0], teacher_id[0], reply_message))
                conn.commit()
            else:
                flash("Notification not found!")
        else:
            flash("Reply message or notification ID missing!")
    
    c.execute("SELECT message, response FROM chat_logs WHERE student_id=? ORDER BY timestamp DESC LIMIT 3", (student[0],))
    chat_history = c.fetchall()
    c.execute("SELECT r.type, r.content, t.name FROM resources r JOIN teachers t ON r.teacher_id=t.id WHERE student_id=?", (student[0],))
    resources = c.fetchall()
    conn.close()

    subjects = ["Math", "DSA", "Java", "Python", "Communication Skills", "Physical Education"]
    academic_data = {subject: {"attendance": 0, "marks": 0, "remarks": "To be updated"} for subject in subjects}
    for row in academics:
        academic_data[row[2]] = {"attendance": row[3], "marks": row[4], "remarks": row[5] or "N/A"}

    chat_response = None
    audio_file = None
    if request.method == 'POST' and 'chat_submit' in request.form:
        user_input = request.form.get('message', '').strip()
        if user_input:
            chat_response, audio_file = get_chatbot_response(student[0], user_input)
        else:
            flash("Chat input cannot be empty!")

    if request.method == 'POST' and 'lang' in request.form:
        lang = request.form.get('lang')
        if lang in ['en', 'hi', 'kn']:
            session['lang'] = lang
            refresh()

    latest_message = notifications[0] if notifications else None
    return render_template('student_home.html', student=student, academic_data=academic_data, notifications=notifications, 
                          chat_history=chat_history, chat_response=chat_response, latest_message=latest_message, 
                          resources=resources, audio_file=audio_file)

# Serve Audio
@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_from_directory(app.config['AUDIO_FOLDER'], filename)

# Admin Route
@app.route('/admin')
def admin_home():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT * FROM teachers")
    teachers = c.fetchall()
    c.execute("SELECT * FROM students")
    students = c.fetchall()
    c.execute("SELECT a.*, s.name AS student_name, t.name AS teacher_name FROM academics a "
              "JOIN students s ON a.student_id=s.id JOIN teachers t ON a.teacher_id=t.id ORDER BY a.timestamp DESC LIMIT 10")
    recent_academics = c.fetchall()
    c.execute("SELECT * FROM hod_logs ORDER BY timestamp DESC LIMIT 10")
    logs = c.fetchall()
    conn.close()
    return render_template('admin.html', teachers=teachers, students=students, recent_academics=recent_academics, logs=logs)

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(AUDIO_FOLDER, exist_ok=True)
    init_db()
    import threading
    threading.Thread(target=hod_terminal, daemon=True).start()
    app.run(debug=True)