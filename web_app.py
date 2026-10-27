"""Local Flask interface for the motorcycle-garage suspicious-event prototype.

This is a coursework prototype. Alerts request human review; they do not prove theft.
"""
from __future__ import annotations

import csv
import os
import re
import sqlite3
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from functools import wraps
from typing import Any

from flask import (
    Flask, abort, flash, redirect, render_template, request, send_from_directory,
    session, url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from main import run_demo

BASE_DIR = Path(__file__).resolve().parent
INSTANCE_DIR = BASE_DIR / "instance"
UPLOAD_DIR = BASE_DIR / "storage" / "uploads"
RUNS_DIR = BASE_DIR / "storage" / "runs"
DB_PATH = INSTANCE_DIR / "app.db"
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "m4v"}
MAX_FILE_BYTES = 500 * 1024 * 1024

for directory in (INSTANCE_DIR, UPLOAD_DIR, RUNS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-change-before-deployment")
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_BYTES
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

ROLES = {
    "admin": "Quản trị viên",
    "security": "Nhân viên an ninh",
    "manager": "Người quản lý",
}
DEMO_ACCOUNTS = [
    ("admin", "admin123", "Quản trị viên", "admin"),
    ("baove", "baove123", "Nhân viên bảo vệ", "security"),
    ("quanly", "quanly123", "Người quản lý", "manager"),
]


def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with db_connect() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                display_name TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('admin','security','manager'))
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS jobs (
                run_id TEXT PRIMARY KEY,
                owner_id INTEGER,
                video_name TEXT NOT NULL,
                source_type TEXT NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                event_count INTEGER NOT NULL DEFAULT 0,
                output_video TEXT,
                error_message TEXT,
                FOREIGN KEY(owner_id) REFERENCES users(id)
            )"""
        )
        count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if count == 0:
            for username, password, display_name, role in DEMO_ACCOUNTS:
                conn.execute(
                    "INSERT INTO users(username,password_hash,display_name,role) VALUES(?,?,?,?)",
                    (username, generate_password_hash(password), display_name, role),
                )


init_db()


def current_user() -> dict[str, Any] | None:
    if "user_id" not in session:
        return None
    return {
        "id": session["user_id"],
        "username": session.get("username", ""),
        "display_name": session.get("display_name", ""),
        "role": session.get("role", "security"),
        "role_label": ROLES.get(session.get("role", "security"), "Người dùng"),
    }


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if current_user() is None:
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


def roles_required(*roles: str):
    def decorator(view):
        @wraps(view)
        @login_required
        def wrapped(*args, **kwargs):
            user = current_user()
            if user is None or user["role"] not in roles:
                flash("Tài khoản của bạn không có quyền thực hiện thao tác này.", "error")
                return redirect(url_for("index"))
            return view(*args, **kwargs)
        return wrapped
    return decorator


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def safe_run_id(run_id: str) -> bool:
    return bool(re.fullmatch(r"[a-f0-9]{12}", run_id))


def get_job(run_id: str):
    if not safe_run_id(run_id):
        return None
    with db_connect() as conn:
        return conn.execute("SELECT * FROM jobs WHERE run_id = ?", (run_id,)).fetchone()


def insert_job(run_id: str, video_name: str, source_type: str, status: str,
               event_count: int = 0, output_video: str | None = None,
               error_message: str | None = None) -> None:
    user = current_user()
    with db_connect() as conn:
        conn.execute(
            """INSERT INTO jobs(run_id,owner_id,video_name,source_type,created_at,status,
                                 event_count,output_video,error_message)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            (run_id, user["id"] if user else None, video_name, source_type,
             datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status, event_count,
             output_video, error_message),
        )


def count_events(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    try:
        with csv_path.open("r", newline="", encoding="utf-8-sig") as stream:
            return sum(1 for _ in csv.DictReader(stream))
    except (OSError, csv.Error):
        return 0


def read_events(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    try:
        with csv_path.open("r", newline="", encoding="utf-8-sig") as stream:
            return list(csv.DictReader(stream))
    except (OSError, csv.Error):
        return []


def get_recent_jobs(limit: int = 8):
    with db_connect() as conn:
        return conn.execute(
            """SELECT j.*, u.display_name AS owner_name
               FROM jobs j LEFT JOIN users u ON u.id=j.owner_id
               ORDER BY j.created_at DESC LIMIT ?""", (limit,)
        ).fetchall()


@app.context_processor
def inject_common_values():
    return {"current_user": current_user(), "role_labels": ROLES}


@app.errorhandler(413)
def too_large(_error):
    flash("Video vượt quá giới hạn 500 MB. Hãy cắt video thành đoạn ngắn hơn.", "error")
    return redirect(url_for("index"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user():
        return redirect(url_for("index"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        with db_connect() as conn:
            user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["display_name"] = user["display_name"]
            session["role"] = user["role"]
            flash(f"Xin chào {user['display_name']}.", "success")
            next_url = request.args.get("next", "")
            # Only allow local relative redirect targets.
            if next_url.startswith("/") and not next_url.startswith("//"):
                return redirect(next_url)
            return redirect(url_for("index"))
        flash("Tên đăng nhập hoặc mật khẩu không đúng.", "error")
    return render_template("login.html")


@app.route("/logout", methods=["POST"])
@login_required
def logout():
    session.clear()
    flash("Bạn đã đăng xuất.", "success")
    return redirect(url_for("login"))


@app.route("/")
@login_required
def index():
    with db_connect() as conn:
        total_jobs = conn.execute("SELECT COUNT(*) FROM jobs WHERE status='done'").fetchone()[0]
        total_alerts = conn.execute("SELECT COALESCE(SUM(event_count),0) FROM jobs WHERE status='done'").fetchone()[0]
    return render_template(
        "index.html", recent_jobs=get_recent_jobs(), total_jobs=total_jobs,
        total_alerts=total_alerts, can_process=current_user()["role"] in ("admin", "security")
    )


@app.route("/process", methods=["POST"])
@roles_required("admin", "security")
def process_video():
    uploaded = request.files.get("video")
    if uploaded is None or not uploaded.filename:
        flash("Bạn chưa chọn video.", "error")
        return redirect(url_for("index"))
    original_name = secure_filename(uploaded.filename)
    if not original_name or not allowed_file(original_name):
        flash("Định dạng không được hỗ trợ. Hãy dùng MP4, AVI, MOV, MKV hoặc M4V.", "error")
        return redirect(url_for("index"))

    run_id = uuid.uuid4().hex[:12]
    input_dir = UPLOAD_DIR / run_id
    output_dir = RUNS_DIR / run_id
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = input_dir / original_name
    uploaded.save(input_path)

    def read_number(name: str, default: float, low: float, high: float) -> float:
        try:
            value = float(request.form.get(name, default))
        except (TypeError, ValueError):
            value = default
        return min(high, max(low, value))

    confidence = read_number("confidence", 0.25, 0.05, 0.95)
    near_px = read_number("near_px", 80, 10, 250)
    alert_seconds = read_number("alert_seconds", 8, 1, 120)
    try:
        max_width = int(request.form.get("max_width", "1280"))
    except ValueError:
        max_width = 1280
    max_width = min(1920, max(320, max_width))
    model_name = request.form.get("model", "yolo11n.pt")
    if model_name not in {"yolo11n.pt", "yolov8n.pt"}:
        model_name = "yolo11n.pt"

    command = [
        sys.executable, str(BASE_DIR / "main.py"),
        "--source", str(input_path),
        "--output-dir", str(output_dir),
        "--model", model_name,
        "--conf", str(confidence),
        "--near-px", str(near_px),
        "--alert-seconds", str(alert_seconds),
        "--max-width", str(max_width),
    ]
    if request.form.get("clahe") == "on":
        command.append("--clahe")
    if request.form.get("denoise") == "on":
        command.append("--denoise")

    try:
        completed = subprocess.run(
            command, cwd=BASE_DIR, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=7200, check=False,
        )
        if completed.returncode == 0:
            video_name = "annotated_video.mp4"
            csv_path = output_dir / "alerts.csv"
            event_count = count_events(csv_path)
            insert_job(run_id, original_name, "video", "done", event_count, video_name)
        else:
            message = (completed.stderr or completed.stdout or "Không có thông tin lỗi.")[-5000:]
            insert_job(run_id, original_name, "video", "error", 0, None, message)
    except subprocess.TimeoutExpired:
        message = "Quá thời gian xử lý 2 giờ. Hãy thử video ngắn hơn hoặc giảm độ phân giải."
        insert_job(run_id, original_name, "video", "error", 0, None, message)
    except OSError as exc:
        message = f"Không thể khởi chạy bộ xử lý: {exc}"
        insert_job(run_id, original_name, "video", "error", 0, None, message)

    return redirect(url_for("result", run_id=run_id))


@app.route("/demo", methods=["POST"])
@roles_required("admin", "security")
def demo():
    run_id = uuid.uuid4().hex[:12]
    output_dir = RUNS_DIR / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        run_demo(output_dir)
        insert_job(run_id, "Video mô phỏng tự sinh", "demo", "done",
                   count_events(output_dir / "alerts.csv"), "demo_annotated.mp4")
    except Exception as exc:  # save a visible failure for local debugging
        insert_job(run_id, "Video mô phỏng tự sinh", "demo", "error", 0, None, str(exc))
    return redirect(url_for("result", run_id=run_id))


@app.route("/result/<run_id>")
@login_required
def result(run_id: str):
    job = get_job(run_id)
    if job is None:
        abort(404)
    output_dir = RUNS_DIR / run_id
    events = read_events(output_dir / "alerts.csv")
    snapshots = sorted(p.name for p in output_dir.glob("alert_*.jpg"))
    return render_template(
        "result.html", job=job, run_id=run_id, events=events, snapshots=snapshots,
        video_exists=bool(job["output_video"] and (output_dir / job["output_video"]).exists()),
        csv_exists=(output_dir / "alerts.csv").exists(),
        demo_frame_exists=(output_dir / "demo_frame.png").exists(),
    )


@app.route("/history")
@login_required
def history():
    return render_template("history.html", jobs=get_recent_jobs(100))


@app.route("/files/<run_id>/<path:filename>")
@login_required
def run_file(run_id: str, filename: str):
    job = get_job(run_id)
    if job is None:
        abort(404)
    directory = RUNS_DIR / run_id
    target = directory / filename
    if not target.is_file():
        abort(404)
    return send_from_directory(directory, filename, as_attachment=filename.lower().endswith(".csv"))


if __name__ == "__main__":
    print("Motorcycle Garage Alert — local coursework prototype")
    print("Open http://127.0.0.1:5000 after the server starts.")
    print("Demo accounts are listed in README.md; change them before any real deployment.")
    app.run(host="127.0.0.1", port=5000, debug=False)
