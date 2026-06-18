"""
Flask web application for the leaderboard system.
"""

import os
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

from config import UPLOADS_DIR, RESULTS_DIR
from models.submission import BenchmarkType
from scoring.engine import ScoringEngine
from leaderboard_manager import LeaderboardManager
from data_handlers.ground_truth import GroundTruthManager


# Initialize Flask app
app = Flask(__name__, template_folder=Path(__file__).parent.parent / "frontend" / "templates", 
            static_folder=Path(__file__).parent.parent / "frontend" / "static")
app.config["UPLOAD_FOLDER"] = str(UPLOADS_DIR)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB max file size

# Initialize managers
scoring_engine = ScoringEngine(use_ollama=False)  # Set to True if Ollama available
leaderboard_manager = LeaderboardManager()
gt_manager = GroundTruthManager()


@app.route("/", methods=["GET"])
def index():
    """Home page."""
    try:
        stats = leaderboard_manager.get_statistics()
        return render_template("index.html", stats=stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/submit", methods=["POST"])
def submit_prediction():
    """Submit predictions for evaluation.
    
    Expected form data:
    - file: CSV or JSON file with predictions
    - model_name: Name of the model
    - benchmark: 'do_you_see_me' or 'minds_eye'
    - task_name: (optional) specific task to evaluate
    """
    try:
        # Validate request
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        if "model_name" not in request.form:
            return jsonify({"error": "model_name required"}), 400

        if "benchmark" not in request.form:
            return jsonify({"error": "benchmark required"}), 400

        file = request.files["file"]
        model_name = request.form["model_name"].strip()
        benchmark_str = request.form["benchmark"].strip()
        task_name = request.form.get("task_name", "").strip() or None

        # Validate file
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not (file.filename.endswith(".csv") or file.filename.endswith(".json")):
            return jsonify({"error": "Only CSV and JSON files are supported"}), 400

        # Validate benchmark
        try:
            if benchmark_str == "do_you_see_me":
                benchmark = BenchmarkType.DO_YOU_SEE_ME
            elif benchmark_str == "minds_eye":
                benchmark = BenchmarkType.MINDS_EYE
            else:
                return jsonify({"error": f"Invalid benchmark: {benchmark_str}"}), 400
        except ValueError:
            return jsonify({"error": f"Invalid benchmark: {benchmark_str}"}), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = UPLOADS_DIR / f"{model_name}_{filename}"
        file.save(str(filepath))

        # Score submission
        try:
            submission_score = scoring_engine.score_submission(
                submission_file=filepath,
                benchmark=benchmark,
                model_name=model_name,
                task_name=task_name,
            )
        except Exception as e:
            filepath.unlink(missing_ok=True)
            return jsonify({"error": f"Scoring failed: {str(e)}"}), 400

        # Save results
        result_file = scoring_engine.save_results(submission_score)

        # Add to leaderboard
        leaderboard_manager.add_submission(submission_score)

        return jsonify({
            "success": True,
            "submission_id": submission_score.submission_id,
            "model_name": submission_score.model_name,
            "benchmark": submission_score.benchmark.value,
            "overall_accuracy": submission_score.overall_accuracy,
            "total_samples": submission_score.total_samples,
            "correct_samples": submission_score.correct_samples,
            "task_results": {
                task: {
                    "accuracy": result.accuracy,
                    "total_samples": result.total_samples,
                    "correct_samples": result.correct_samples,
                    "std_dev": result.std_dev,
                }
                for task, result in submission_score.task_results.items()
            },
        }), 200

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route("/api/leaderboard", methods=["GET"])
def get_leaderboard():
    """Get leaderboard rankings.
    
    Query parameters:
    - benchmark: 'do_you_see_me' or 'minds_eye' (optional)
    - limit: number of top results (default: 50)
    - task: specific task to filter by (optional)
    """
    try:
        benchmark_str = request.args.get("benchmark", "").strip()
        limit = int(request.args.get("limit", 50))
        task_name = request.args.get("task", "").strip() or None

        # Parse benchmark
        benchmark = None
        if benchmark_str:
            if benchmark_str == "do_you_see_me":
                benchmark = BenchmarkType.DO_YOU_SEE_ME
            elif benchmark_str == "minds_eye":
                benchmark = BenchmarkType.MINDS_EYE
            else:
                return jsonify({"error": f"Invalid benchmark: {benchmark_str}"}), 400

        # Get leaderboard
        if task_name:
            entries = leaderboard_manager.get_leaderboard_by_task(
                task_name, benchmark=benchmark, limit=limit
            )
        else:
            entries = leaderboard_manager.get_leaderboard(
                benchmark=benchmark, limit=limit
            )

        return jsonify({
            "leaderboard": [entry.to_dict() for entry in entries],
            "count": len(entries),
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/submission/<submission_id>", methods=["GET"])
def get_submission_details(submission_id):
    """Get detailed submission information.
    
    Args:
        submission_id: Submission ID
    """
    try:
        details = leaderboard_manager.get_submission_details(submission_id)
        if details is None:
            return jsonify({"error": "Submission not found"}), 404

        return jsonify(details), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/model/<model_name>", methods=["GET"])
def get_model_submissions(model_name):
    """Get all submissions for a model.
    
    Args:
        model_name: Model name
    """
    try:
        submissions = leaderboard_manager.get_model_submissions(model_name)
        return jsonify({
            "model_name": model_name,
            "submissions": submissions,
            "count": len(submissions),
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/tasks", methods=["GET"])
def get_available_tasks():
    """Get list of available tasks.
    
    Query parameters:
    - benchmark: 'do_you_see_me' or 'minds_eye' (optional)
    """
    try:
        benchmark_str = request.args.get("benchmark", "").strip()

        tasks = gt_manager.list_available_tasks()

        if benchmark_str == "do_you_see_me":
            tasks = {"do_you_see_me": tasks["do_you_see_me"]}
        elif benchmark_str == "minds_eye":
            tasks = {"minds_eye": tasks["minds_eye"]}
        elif benchmark_str and benchmark_str not in ["do_you_see_me", "minds_eye"]:
            return jsonify({"error": f"Invalid benchmark: {benchmark_str}"}), 400

        return jsonify(tasks), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/statistics", methods=["GET"])
def get_statistics():
    """Get leaderboard statistics.
    
    Query parameters:
    - benchmark: 'do_you_see_me' or 'minds_eye' (optional)
    """
    try:
        benchmark_str = request.args.get("benchmark", "").strip()

        benchmark = None
        if benchmark_str:
            if benchmark_str == "do_you_see_me":
                benchmark = BenchmarkType.DO_YOU_SEE_ME
            elif benchmark_str == "minds_eye":
                benchmark = BenchmarkType.MINDS_EYE
            else:
                return jsonify({"error": f"Invalid benchmark: {benchmark_str}"}), 400

        stats = leaderboard_manager.get_statistics(benchmark=benchmark)
        return jsonify(stats), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors."""
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    # Development server
    debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=5000, debug=debug)
