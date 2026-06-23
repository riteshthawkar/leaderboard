"""
Production-ready Flask web application for the leaderboard system.
"""

import os
import re
import secrets
import sys
import json
import logging
import uuid
from pathlib import Path
from functools import wraps
from datetime import datetime, timedelta

# Ensure the backend package directory is importable when running this file
# directly (e.g. `python backend/web/app.py`) so that `from config import ...`
# and the other top-level module imports below resolve correctly.
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from flask import Flask, request, jsonify, render_template, g, send_file
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_httpauth import HTTPTokenAuth
from werkzeug.exceptions import RequestEntityTooLarge
import requests

from config import (
    UPLOADS_DIR, RESULTS_DIR,
    TASKS, SECTIONS, SPATIAL_DATASETS, SPATIAL_MANIFEST_FILE, NO_IMAGE_PLUS_OPTION,
    LAYER_LABELS, VCI_LAYER_WEIGHTS, EVAL_CONDITIONS, GRADING,
)
from models.submission import BenchmarkType
from scoring.engine import ScoringEngine
from auth_db import init_db as init_auth_db, register_user, login_user, get_username_by_token
from scoring.task_scorer import TaskScorer
from leaderboard_manager import LeaderboardManager
from leaderboard_store import LeaderboardStore
from data_handlers.ground_truth import GroundTruthManager
from request_models import SubmissionRequest, LeaderboardRequest, ErrorResponse, HealthCheckResponse
from file_security import FileSecurityValidator
from constants import (
    ERROR_INVALID_BENCHMARK,
    SUBMISSIONS_PER_HOUR,
    SUBMISSIONS_PER_DAY,
    DEFAULT_LEADERBOARD_LIMIT,
)
from logging_config import logger

# Configure Flask app
# __file__ is backend/web/app.py, so the project root (which contains the
# `frontend/` directory) is three levels up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "frontend" / "templates"),
    static_folder=str(PROJECT_ROOT / "frontend" / "static")
)

# Secret key – required for Flask internals (CSRF helpers, signed cookies, etc.)
# Generate with: python -c "import secrets; print(secrets.token_hex(32))"
_secret = os.getenv("SECRET_KEY", "")
if not _secret:
    import warnings
    warnings.warn(
        "SECRET_KEY not set in environment. Using a random key — sessions will "
        "not survive server restarts. Set SECRET_KEY in your .env for production.",
        stacklevel=1,
    )
    _secret = secrets.token_hex(32)
app.secret_key = _secret

# CORS configuration
CORS(app, resources={
    r"/api/*": {
        "origins": os.getenv("CORS_ORIGINS", "*").split(","),
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# File upload configuration
app.config["UPLOAD_FOLDER"] = str(UPLOADS_DIR)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri=os.getenv("LIMITER_STORAGE_URI", "memory://")
)

# Authentication
auth = HTTPTokenAuth('Bearer')

def get_token_identity():
    """Rate-limit key: resolve token to username if possible, else fall back to IP."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        username = get_username_by_token(token)
        if username:
            return "user:" + username
        return "token:" + token
    return "ip:" + get_remote_address()

# Initialize managers
scoring_engine = ScoringEngine(use_ollama=False)
leaderboard_manager = LeaderboardManager()
gt_manager = GroundTruthManager()

# Three-task Visual Cognition / Spatial managers
task_scorers = {tid: TaskScorer(tid) for tid in TASKS}
leaderboard_store = LeaderboardStore()

# Initialise user auth DB
init_auth_db()

# Generate request ID for tracking
@app.before_request
def before_request():
    """Generate request ID and log incoming request."""
    g.request_id = str(uuid.uuid4())
    g.start_time = datetime.utcnow()
    
    # Log incoming request
    logger.debug(
        f"Incoming {request.method} {request.path}",
        extra={
            "request_id": g.request_id,
            "remote_addr": request.remote_addr,
            "user_agent": request.user_agent.string
        }
    )

@app.after_request
def after_request(response):
    """Log response and add headers."""
    if hasattr(g, 'start_time'):
        duration = (datetime.utcnow() - g.start_time).total_seconds()
        logger.debug(
            f"Response {response.status_code} ({duration:.3f}s)",
            extra={
                "request_id": g.request_id,
                "status_code": response.status_code,
                "duration_ms": duration * 1000
            }
        )
    
    # Add security headers
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
    
    return response

@auth.verify_token
def verify_token(token):
    """Verify API token: check user DB first, fall back to env API_TOKENS for admin."""
    # Check registered users
    username = get_username_by_token(token)
    if username:
        g.user_id = username
        return True
    # Fallback: static admin tokens from environment
    valid_tokens = [t for t in os.getenv("API_TOKENS", "").split(",") if t]
    if token in valid_tokens:
        g.user_id = token
        return True
    logger.warning(f"Invalid token attempted from {request.remote_addr}")
    return False

@auth.error_handler
def auth_error(status=401):
    """Return a JSON 401 so API clients (and the web UI) get a parseable error."""
    return jsonify({
        "error": "Authentication required",
        "request_id": getattr(g, 'request_id', None)
    }), status

# Error handlers
@app.errorhandler(400)
def bad_request(error):
    """Handle 400 errors."""
    logger.warning(f"Bad request: {error}")
    return jsonify({
        "error": "Bad request",
        "request_id": getattr(g, 'request_id', None)
    }), 400

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        "error": "Not found",
        "request_id": getattr(g, 'request_id', None)
    }), 404

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle 413 errors (file too large)."""
    logger.warning(f"Request entity too large from {request.remote_addr}")
    return jsonify({
        "error": "File too large",
        "request_id": getattr(g, 'request_id', None)
    }), 413

@app.errorhandler(429)
def ratelimit_handler(e):
    """Handle rate limiting."""
    logger.warning(f"Rate limit exceeded for {request.remote_addr}: {e.description}")
    return jsonify({
        "error": "Rate limit exceeded",
        "message": e.description,
        "request_id": getattr(g, 'request_id', None)
    }), 429

@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors."""
    request_id = getattr(g, 'request_id', None)
    logger.error(f"Server error: {error}", extra={"request_id": request_id}, exc_info=True)
    return jsonify({
        "error": "Internal server error",
        "request_id": request_id
    }), 500

# Routes
# ------------------------------------------------------------------ auth
@app.route("/api/auth/register", methods=["POST"])
@limiter.limit("5 per hour")
def auth_register():
    """Register a new user. Returns an API token on success."""
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip().lower()
    password = str(data.get("password") or "")
    if not username or not password:
        return jsonify({"error": "username and password are required"}), 400
    if len(username) < 3 or len(username) > 40:
        return jsonify({"error": "username must be 3–40 characters"}), 400
    if not re.fullmatch(r'[a-z0-9_-]+', username):
        return jsonify({"error": "username may only contain letters, numbers, hyphens and underscores"}), 400
    if len(password) < 6:
        return jsonify({"error": "password must be at least 6 characters"}), 400
    token = register_user(username, password)
    if token is None:
        return jsonify({"error": "Username already taken"}), 409
    return jsonify({"username": username, "api_token": token}), 201


@app.route("/api/auth/login", methods=["POST"])
@limiter.limit("10 per hour")
def auth_login():
    """Log in an existing user. Returns their API token."""
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip().lower()
    password = str(data.get("password") or "")
    if not username or not password:
        return jsonify({"error": "username and password are required"}), 400
    token = login_user(username, password)
    if token is None:
        return jsonify({"error": "Invalid username or password"}), 401
    return jsonify({"username": username, "api_token": token}), 200


@app.route("/", methods=["GET"])
def index():
    """Combined home / landing page."""
    try:
        return render_template("home.html")
    except Exception as e:
        logger.error(f"Error loading index: {e}", exc_info=True)
        return jsonify({"error": "Failed to load page"}), 500


@app.route("/benchmarks/do-you-see-me", methods=["GET"])
def page_dysm():
    """Do-You-See-Me benchmark page."""
    return render_template("benchmark_dysm.html")


@app.route("/benchmarks/minds-eye", methods=["GET"])
def page_minds_eye():
    """Mind's-Eye benchmark page."""
    return render_template("benchmark_minds_eye.html")


@app.route("/benchmarks/spatial", methods=["GET"])
def page_spatial():
    """Spatial reasoning benchmark page."""
    return render_template("benchmark_spatial.html")


@app.route("/leaderboard", methods=["GET"])
def page_leaderboard():
    """Model rankings page (Visual Cognition + Spatial tracks)."""
    return render_template("leaderboard.html")


@app.route("/login", methods=["GET"])
def page_login():
    """Login / register page."""
    return render_template("login.html")


@app.route("/submit", methods=["GET"])
def page_submit():
    """Submission page."""
    return render_template("submit.html")

@app.route("/api/health", methods=["GET"])
@limiter.limit("60 per minute")
def health_check():
    """Health check endpoint for monitoring."""
    try:
        components = {}
        
        # Check leaderboard manager
        try:
            leaderboard_manager.get_leaderboard(limit=1)
            components["database"] = "healthy"
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            components["database"] = "unhealthy"
        
        # Check ground truth loading
        try:
            gt_manager.list_available_tasks()
            components["ground_truth"] = "healthy"
        except Exception as e:
            logger.error(f"Ground truth health check failed: {e}")
            components["ground_truth"] = "unhealthy"
        
        # Determine overall status
        overall_status = "healthy" if all(v == "healthy" for v in components.values()) else "degraded"
        
        response = HealthCheckResponse(
            status=overall_status,
            timestamp=datetime.utcnow().isoformat(),
            components=components
        )
        
        status_code = 200 if overall_status == "healthy" else 503
        return jsonify(response.dict()), status_code
        
    except Exception as e:
        logger.error(f"Health check error: {e}", exc_info=True)
        return jsonify({
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": "Health check failed"
        }), 500

@app.route("/api/submit", methods=["POST"])
@limiter.limit(f"{SUBMISSIONS_PER_HOUR} per hour", key_func=get_token_identity)
@limiter.limit(f"{SUBMISSIONS_PER_DAY} per day", key_func=get_token_identity)
@auth.login_required
def submit_prediction():
    """Submit predictions for evaluation (requires authentication)."""
    request_id = getattr(g, 'request_id', None)
    
    try:
        logger.info("Submission request received", extra={"request_id": request_id})
        
        # Validate request
        if "file" not in request.files:
            logger.warning("No file in request", extra={"request_id": request_id})
            return jsonify({"error": "No file provided"}), 400

        if "model_name" not in request.form:
            logger.warning("No model_name in request", extra={"request_id": request_id})
            return jsonify({"error": "model_name required"}), 400

        if "benchmark" not in request.form:
            logger.warning("No benchmark in request", extra={"request_id": request_id})
            return jsonify({"error": "benchmark required"}), 400

        file = request.files["file"]
        
        # Validate file upload
        is_valid, error_msg, safe_filename = FileSecurityValidator.validate_and_secure_upload(
            file.stream, file.filename
        )
        if not is_valid:
            logger.warning(f"File validation failed: {error_msg}", extra={"request_id": request_id})
            return jsonify({"error": error_msg}), 400
        
        # Validate request data using Pydantic
        try:
            submission_req = SubmissionRequest(
                model_name=request.form["model_name"].strip(),
                benchmark=request.form["benchmark"].strip(),
                task_name=request.form.get("task_name", "").strip() or None
            )
        except Exception as e:
            logger.warning(f"Request validation failed: {e}", extra={"request_id": request_id})
            return jsonify({"error": f"Invalid request: {str(e)}"}), 400
        
        # Save uploaded file with safe name
        filepath = UPLOADS_DIR / safe_filename
        try:
            file.save(str(filepath))
            logger.info(f"File saved: {safe_filename}", extra={"request_id": request_id})
        except Exception as e:
            logger.error(f"Failed to save file: {e}", extra={"request_id": request_id}, exc_info=True)
            return jsonify({"error": "Failed to save file"}), 500

        # Score submission
        try:
            logger.info(
                "Scoring submission",
                extra={
                    "request_id": request_id,
                    "model_name": submission_req.model_name,
                    "benchmark": submission_req.benchmark.value
                }
            )
            
            submission_score = scoring_engine.score_submission(
                submission_file=filepath,
                benchmark=BenchmarkType(submission_req.benchmark.value),
                model_name=submission_req.model_name,
                task_name=submission_req.task_name,
            )
            
            logger.info(
                f"Scoring completed: {submission_score.overall_accuracy:.4f}",
                extra={
                    "request_id": request_id,
                    "submission_id": submission_score.submission_id,
                    "accuracy": submission_score.overall_accuracy
                }
            )
            
        except Exception as e:
            logger.error(f"Scoring failed: {e}", extra={"request_id": request_id}, exc_info=True)
            try:
                filepath.unlink(missing_ok=True)
            except:
                pass
            return jsonify({"error": f"Scoring failed: {str(e)}"}), 400

        # Save results
        try:
            result_file = scoring_engine.save_results(submission_score)
            logger.info(f"Results saved: {result_file}", extra={"request_id": request_id})
        except Exception as e:
            logger.error(f"Failed to save results: {e}", extra={"request_id": request_id}, exc_info=True)

        # Add to leaderboard
        try:
            leaderboard_manager.add_submission(submission_score)
            logger.info("Submission added to leaderboard", extra={"request_id": request_id})
        except Exception as e:
            logger.error(f"Failed to add to leaderboard: {e}", extra={"request_id": request_id}, exc_info=True)

        response = {
            "success": True,
            "submission_id": submission_score.submission_id,
            "model_name": submission_score.model_name,
            "benchmark": submission_score.benchmark.value,
            "overall_accuracy": round(submission_score.overall_accuracy, 4),
            "total_samples": submission_score.total_samples,
            "correct_samples": submission_score.correct_samples,
            "task_results": {
                task: {
                    "accuracy": round(result.accuracy, 4),
                    "total_samples": result.total_samples,
                    "correct_samples": result.correct_samples,
                    "std_dev": round(result.std_dev, 4),
                }
                for task, result in submission_score.task_results.items()
            },
            "request_id": request_id,
        }
        
        logger.info("Submission successful", extra={"request_id": request_id})
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Unexpected error in submission: {e}", extra={"request_id": request_id}, exc_info=True)
        return jsonify({
            "error": "Server error processing submission",
            "request_id": request_id
        }), 500

@app.route("/api/leaderboard", methods=["GET"])
@limiter.limit("60 per minute")
def get_leaderboard():
    """Get leaderboard rankings."""
    request_id = getattr(g, 'request_id', None)
    
    try:
        # Validate request using Pydantic
        try:
            lb_req = LeaderboardRequest(
                benchmark=request.args.get("benchmark", "").strip() or None,
                task=request.args.get("task", "").strip() or None,
                limit=int(request.args.get("limit", DEFAULT_LEADERBOARD_LIMIT))
            )
        except Exception as e:
            logger.warning(f"Invalid leaderboard request: {e}", extra={"request_id": request_id})
            return jsonify({"error": f"Invalid request: {str(e)}"}), 400

        # Get leaderboard
        if lb_req.task:
            entries = leaderboard_manager.get_leaderboard_by_task(
                lb_req.task, benchmark=lb_req.benchmark, limit=lb_req.limit
            )
            logger.info(f"Task leaderboard retrieved: {lb_req.task}", extra={"request_id": request_id})
        else:
            entries = leaderboard_manager.get_leaderboard(
                benchmark=lb_req.benchmark, limit=lb_req.limit
            )
            logger.info(f"Leaderboard retrieved: {len(entries)} entries", extra={"request_id": request_id})

        return jsonify({
            "leaderboard": [entry.to_dict() for entry in entries],
            "count": len(entries),
            "request_id": request_id,
        }), 200

    except Exception as e:
        logger.error(f"Leaderboard error: {e}", extra={"request_id": request_id}, exc_info=True)
        return jsonify({
            "error": "Failed to retrieve leaderboard",
            "request_id": request_id
        }), 500

@app.route("/api/submission/<submission_id>", methods=["GET"])
@limiter.limit("60 per minute")
def get_submission_details(submission_id):
    """Get detailed submission information."""
    request_id = getattr(g, 'request_id', None)
    
    try:
        # Validate submission_id
        if not submission_id or len(submission_id) > 255:
            return jsonify({"error": "Invalid submission ID"}), 400
        
        logger.info(f"Retrieving submission details: {submission_id}", extra={"request_id": request_id})
        
        details = leaderboard_manager.get_submission_details(submission_id)
        if details is None:
            logger.warning(f"Submission not found: {submission_id}", extra={"request_id": request_id})
            return jsonify({"error": "Submission not found"}), 404

        return jsonify({**details, "request_id": request_id}), 200

    except Exception as e:
        logger.error(f"Error retrieving submission: {e}", extra={"request_id": request_id}, exc_info=True)
        return jsonify({
            "error": "Failed to retrieve submission",
            "request_id": request_id
        }), 500

@app.route("/api/tasks", methods=["GET"])
@limiter.limit("60 per minute")
def get_available_tasks():
    """Get list of available tasks."""
    request_id = getattr(g, 'request_id', None)
    
    try:
        benchmark_str = request.args.get("benchmark", "").strip()

        tasks = gt_manager.list_available_tasks()
        logger.info(f"Available tasks retrieved", extra={"request_id": request_id})

        if benchmark_str == "do_you_see_me":
            tasks = {"do_you_see_me": tasks["do_you_see_me"]}
        elif benchmark_str == "minds_eye":
            tasks = {"minds_eye": tasks["minds_eye"]}
        elif benchmark_str and benchmark_str not in ["do_you_see_me", "minds_eye"]:
            logger.warning(f"Invalid benchmark: {benchmark_str}", extra={"request_id": request_id})
            return jsonify({"error": ERROR_INVALID_BENCHMARK}), 400

        return jsonify({**tasks, "request_id": request_id}), 200

    except Exception as e:
        logger.error(f"Error retrieving tasks: {e}", extra={"request_id": request_id}, exc_info=True)
        return jsonify({
            "error": "Failed to retrieve tasks",
            "request_id": request_id
        }), 500

@app.route("/api/statistics", methods=["GET"])
@limiter.limit("60 per minute")
def get_statistics():
    """Get leaderboard statistics."""
    request_id = getattr(g, 'request_id', None)
    
    try:
        benchmark_str = request.args.get("benchmark", "").strip()

        benchmark = None
        if benchmark_str:
            if benchmark_str == "do_you_see_me":
                benchmark = BenchmarkType.DO_YOU_SEE_ME
            elif benchmark_str == "minds_eye":
                benchmark = BenchmarkType.MINDS_EYE
            else:
                logger.warning(f"Invalid benchmark: {benchmark_str}", extra={"request_id": request_id})
                return jsonify({"error": ERROR_INVALID_BENCHMARK}), 400

        stats = leaderboard_manager.get_statistics(benchmark=benchmark)
        logger.info("Statistics retrieved", extra={"request_id": request_id})
        
        return jsonify({**stats, "request_id": request_id}), 200

    except Exception as e:
        logger.error(f"Error retrieving statistics: {e}", extra={"request_id": request_id}, exc_info=True)
        return jsonify({
            "error": "Failed to retrieve statistics",
            "request_id": request_id
        }), 500

# ---------------------------------------------------------------------------
# Three-task Visual Cognition / Spatial Reasoning endpoints
# ---------------------------------------------------------------------------

def _task_or_404(task_id):
    return TASKS.get(task_id)


@app.route("/api/sections", methods=["GET"])
@limiter.limit("60 per minute")
def api_sections():
    """UI layout: the two sections, their tasks, layers and VCI weights."""
    request_id = getattr(g, "request_id", None)
    try:
        sections = []
        for sec in SECTIONS.values():
            sections.append({
                "id": sec["id"],
                "label": sec["label"],
                "primary_metric": sec["primary_metric"],
                "tasks": [
                    {
                        "task_id": TASKS[t]["task_id"],
                        "label": TASKS[t]["label"],
                        "layer": TASKS[t]["layer"],
                        "order": TASKS[t]["order"],
                        "supports_diagnostics": TASKS[t]["supports_diagnostics"],
                        "description": TASKS[t]["description"],
                    }
                    for t in sec["tasks"]
                ],
            })
        return jsonify({
            "sections": sections,
            "layer_labels": LAYER_LABELS,
            "vci_weights": VCI_LAYER_WEIGHTS,
            "eval_conditions": EVAL_CONDITIONS,
            "request_id": request_id,
        }), 200
    except Exception as e:
        logger.error(f"Sections error: {e}", extra={"request_id": request_id}, exc_info=True)
        return jsonify({"error": "Failed to load sections"}), 500


@app.route("/api/tasks/<task_id>/info", methods=["GET"])
@limiter.limit("60 per minute")
def task_info(task_id):
    """Metadata + sample count for a single task."""
    request_id = getattr(g, "request_id", None)
    task = _task_or_404(task_id)
    if not task:
        return jsonify({"error": "Unknown task"}), 404
    try:
        total = 0
        qfile = task["paths"]["questions"]
        if qfile.exists():
            with open(qfile, "r", encoding="utf-8") as f:
                total = json.load(f).get("total_samples", 0)
        info = {
            "task_id": task["task_id"],
            "label": task["label"],
            "section": task["section"],
            "layer": task["layer"],
            "group_by": task["group_by"],
            "supports_diagnostics": task["supports_diagnostics"],
            "description": task["description"],
            "total_samples": total,
        }
        # Advertise how this task is graded (which paper pipeline + judge model)
        # so the UI can show it even before any submission. Never expose keys.
        gcfg = GRADING.get(task_id, {})
        if gcfg:
            info["grading"] = {
                "method": gcfg.get("method"),
                "judge_model": gcfg.get("judge_model"),
                "paper": gcfg.get("paper"),
                "random_baseline": gcfg.get("random_baseline"),
            }
        if task_id == "spatial":
            info["datasets"] = SPATIAL_DATASETS
            info["conditions"] = EVAL_CONDITIONS
            info["no_image_plus_option"] = NO_IMAGE_PLUS_OPTION
        return jsonify({**info, "request_id": request_id}), 200
    except Exception as e:
        logger.error(f"Task info error: {e}", extra={"request_id": request_id}, exc_info=True)
        return jsonify({"error": "Failed to load task info"}), 500


@app.route("/api/tasks/<task_id>/questions", methods=["GET"])
@limiter.limit("30 per minute")
def task_questions(task_id):
    """Download a task's public sample set (no answers)."""
    task = _task_or_404(task_id)
    if not task:
        return jsonify({"error": "Unknown task"}), 404
    qfile = task["paths"]["questions"]
    if not qfile.exists():
        return jsonify({"error": "Task not built yet"}), 404
    return send_file(str(qfile), as_attachment=True,
                     download_name=f"{task_id}_questions.json", mimetype="application/json")


@app.route("/api/tasks/<task_id>/template.<fmt>", methods=["GET"])
@limiter.limit("30 per minute")
def task_template(task_id, fmt):
    """Download a task's submission template (json or csv)."""
    task = _task_or_404(task_id)
    if not task:
        return jsonify({"error": "Unknown task"}), 404
    if fmt == "json" and task["paths"]["template_json"].exists():
        return send_file(str(task["paths"]["template_json"]), as_attachment=True,
                         download_name=f"{task_id}_template.json", mimetype="application/json")
    if fmt == "csv" and task["paths"]["template_csv"].exists():
        return send_file(str(task["paths"]["template_csv"]), as_attachment=True,
                         download_name=f"{task_id}_template.csv", mimetype="text/csv")
    return jsonify({"error": "Template not found"}), 404


@app.route("/api/spatial/manifest", methods=["GET"])
@limiter.limit("30 per minute")
def spatial_manifest():
    """Download the Task-3 dataset manifest (the public spec for the harness)."""
    if not SPATIAL_MANIFEST_FILE.exists():
        return jsonify({"error": "Manifest not built yet"}), 404
    return send_file(str(SPATIAL_MANIFEST_FILE), as_attachment=True,
                     download_name="spatial_manifest.json", mimetype="application/json")


@app.route("/api/tasks/<task_id>/submit", methods=["POST"])
@limiter.limit(f"{SUBMISSIONS_PER_HOUR} per hour", key_func=get_token_identity)
@limiter.limit(f"{SUBMISSIONS_PER_DAY} per day", key_func=get_token_identity)
@auth.login_required
def submit_task(task_id):
    """Submit one task's predictions, score them, and update the model entry."""
    request_id = getattr(g, "request_id", None)
    task = _task_or_404(task_id)
    if not task:
        return jsonify({"error": "Unknown task"}), 404
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        if "model_name" not in request.form:
            return jsonify({"error": "model_name required"}), 400

        file = request.files["file"]
        is_valid, error_msg, safe_filename = FileSecurityValidator.validate_and_secure_upload(
            file.stream, file.filename
        )
        if not is_valid:
            return jsonify({"error": error_msg}), 400

        model_name = request.form["model_name"].strip()
        if not model_name or len(model_name) > 255:
            return jsonify({"error": "Invalid model_name"}), 400

        filepath = UPLOADS_DIR / safe_filename
        file.save(str(filepath))

        try:
            score = task_scorers[task_id].score(filepath, model_name=model_name)
        except (ValueError, FileNotFoundError) as e:
            logger.warning(f"Task scoring failed: {e}", extra={"request_id": request_id})
            try:
                filepath.unlink(missing_ok=True)
            except Exception:
                pass
            return jsonify({"error": f"Scoring failed: {str(e)}"}), 400

        record = leaderboard_store.add_result(score, submitted_by=getattr(g, 'user_id', None))
        logger.info(
            f"Task '{task_id}' scored for {model_name}: acc={score.accuracy:.4f}",
            extra={"request_id": request_id},
        )
        return jsonify({**record, "success": True, "request_id": request_id}), 200
    except PermissionError as e:
        return jsonify({"error": str(e)}), 403
    except Exception as e:
        logger.error(f"Task submission error: {e}", extra={"request_id": request_id}, exc_info=True)
        return jsonify({"error": "Server error processing submission", "request_id": request_id}), 500


@app.route("/api/leaderboard/visual-cognition", methods=["GET"])
@limiter.limit("60 per minute")
def leaderboard_visual_cognition():
    """Combined Do-You-See-Me + Mind's-Eye ranking (VCI)."""
    request_id = getattr(g, "request_id", None)
    try:
        limit = int(request.args.get("limit", 100))
        rows = leaderboard_store.visual_cognition_leaderboard(limit=limit)
        return jsonify({"leaderboard": rows, "count": len(rows),
                        "request_id": request_id}), 200
    except Exception as e:
        logger.error(f"VC leaderboard error: {e}", extra={"request_id": request_id}, exc_info=True)
        return jsonify({"error": "Failed to retrieve leaderboard"}), 500


@app.route("/api/leaderboard/spatial", methods=["GET"])
@limiter.limit("60 per minute")
def leaderboard_spatial():
    """Task-3 spatial ranking with robustness diagnostics."""
    request_id = getattr(g, "request_id", None)
    try:
        limit = int(request.args.get("limit", 100))
        rows = leaderboard_store.spatial_leaderboard(limit=limit)
        return jsonify({"leaderboard": rows, "count": len(rows),
                        "request_id": request_id}), 200
    except Exception as e:
        logger.error(f"Spatial leaderboard error: {e}", extra={"request_id": request_id}, exc_info=True)
        return jsonify({"error": "Failed to retrieve leaderboard"}), 500


@app.route("/api/statistics/overview", methods=["GET"])
@limiter.limit("60 per minute")
def statistics_overview():
    request_id = getattr(g, "request_id", None)
    try:
        return jsonify({**leaderboard_store.statistics(), "request_id": request_id}), 200
    except Exception as e:
        logger.error(f"Overview statistics error: {e}", extra={"request_id": request_id}, exc_info=True)
        return jsonify({"error": "Failed to retrieve statistics"}), 500


@app.route("/api/model/<path:model_name>/report", methods=["GET"])
@limiter.limit("60 per minute")
def model_report(model_name):
    """Full per-model report across all three tasks."""
    request_id = getattr(g, "request_id", None)
    try:
        if not model_name or len(model_name) > 255:
            return jsonify({"error": "Invalid model name"}), 400
        report = leaderboard_store.get_model(model_name)
        if report is None:
            return jsonify({"error": "Model not found"}), 404
        return jsonify({**report, "request_id": request_id}), 200
    except Exception as e:
        logger.error(f"Model report error: {e}", extra={"request_id": request_id}, exc_info=True)
        return jsonify({"error": "Failed to retrieve report"}), 500

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5000"))
    use_dev_server = os.getenv("FLASK_DEBUG", "").lower() in ("1", "true", "yes")

    if use_dev_server:
        # Development server (auto-reload, debugger). Not for production.
        logger.info(f"Starting Flask development server on {host}:{port}")
        app.run(host=host, port=port, debug=True, use_reloader=True)
    else:
        # Production-grade WSGI server. Waitress works on Windows (unlike
        # Gunicorn, which is Unix-only).
        try:
            from waitress import serve
        except ImportError:
            logger.warning(
                "waitress not installed; falling back to the Flask development "
                "server. Install it with `pip install waitress` for production."
            )
            app.run(host=host, port=port, debug=False)
        else:
            logger.info(f"Starting Waitress production server on {host}:{port}")
            serve(app, host=host, port=port, threads=8)
