"""
Test script to validate the Combined Leaderboard system.
"""

import sys
from pathlib import Path
import json

# Add backend to path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

from models.submission import BenchmarkType, PredictionScore, TaskResults
from data_handlers.ground_truth import GroundTruthManager
from data_handlers.submission import SubmissionParser
from utils.answer_extractor import OptionExtractor, AnswerComparator
from scoring.engine import ScoringEngine
from leaderboard_manager import LeaderboardManager


def test_option_extraction():
    """Test option extraction functionality."""
    print("Testing option extraction...")
    
    test_cases = [
        ("The answer is (C)", "C"),
        ("Option B is correct", "B"),
        ("(a)", "A"),
        ("I think it's D", "D"),
    ]
    
    for text, expected in test_cases:
        result = OptionExtractor.extract_option(text)
        assert result == expected, f"Expected {expected}, got {result} for '{text}'"
    
    print("✓ Option extraction tests passed")


def test_answer_comparison():
    """Test answer comparison."""
    print("Testing answer comparison...")
    
    test_cases = [
        ("A", "Option A", True),
        ("(c)", "C", True),
        ("1", "1", True),
        ("cat", "Cat", True),
        ("B", "A", False),
    ]
    
    for gt, pred, expected in test_cases:
        is_correct, _ = AnswerComparator.compare_answers(gt, pred)
        assert is_correct == expected, f"Expected {expected} for {gt} vs {pred}"
    
    print("✓ Answer comparison tests passed")


def test_ground_truth_loading():
    """Test ground truth loading."""
    print("Testing ground truth loading...")
    
    try:
        manager = GroundTruthManager()
        tasks = manager.list_available_tasks()
        
        print(f"  Available Mind's-Eye tasks: {len(tasks.get('minds_eye', []))}")
        print(f"  Available Do-You-See-Me tasks: {len(tasks.get('do_you_see_me', []))}")
        
        if tasks['minds_eye']:
            print(f"    Sample Mind's-Eye tasks: {tasks['minds_eye'][:3]}")
        if tasks['do_you_see_me']:
            print(f"    Sample Do-You-See-Me tasks: {tasks['do_you_see_me'][:3]}")
        
        print("✓ Ground truth loading test passed")
    except Exception as e:
        print(f"✗ Ground truth loading failed: {e}")


def test_submission_parsing():
    """Test submission parsing."""
    print("Testing submission parsing...")
    
    # Create test CSV file
    import tempfile
    import csv
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_name', 'task_name', 'prediction'])
        writer.writerow(['img_0.png', 'task_1', 'A'])
        writer.writerow(['img_1.png', 'task_1', 'B'])
        csv_file = f.name
    
    try:
        predictions = SubmissionParser.parse_csv(Path(csv_file))
        assert 'task_1' in predictions
        assert len(predictions['task_1']) == 2
        print("✓ CSV parsing test passed")
    finally:
        Path(csv_file).unlink()
    
    # Create test JSON file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({
            'task_1': {
                'img_0.png': 'A',
                'img_1.png': 'B'
            }
        }, f)
        json_file = f.name
    
    try:
        predictions = SubmissionParser.parse_json(Path(json_file))
        assert 'task_1' in predictions
        assert len(predictions['task_1']) == 2
        print("✓ JSON parsing test passed")
    finally:
        Path(json_file).unlink()


def test_scoring_engine():
    """Test scoring engine initialization."""
    print("Testing scoring engine...")
    
    try:
        engine = ScoringEngine(use_ollama=False)
        print("✓ Scoring engine initialized successfully")
    except Exception as e:
        print(f"✗ Scoring engine initialization failed: {e}")


def test_leaderboard_manager():
    """Test leaderboard manager."""
    print("Testing leaderboard manager...")
    
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        lb_file = Path(tmpdir) / "leaderboard.json"
        manager = LeaderboardManager(leaderboard_file=lb_file)
        
        # Get empty leaderboard
        entries = manager.get_leaderboard()
        assert len(entries) == 0, "Leaderboard should be empty"
        
        # Get statistics
        stats = manager.get_statistics()
        assert stats['total_submissions'] == 0
        
        print("✓ Leaderboard manager test passed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*50)
    print("Combined Leaderboard - System Tests")
    print("="*50 + "\n")
    
    try:
        test_option_extraction()
        test_answer_comparison()
        test_ground_truth_loading()
        test_submission_parsing()
        test_scoring_engine()
        test_leaderboard_manager()
        
        print("\n" + "="*50)
        print("✓ All tests passed!")
        print("="*50 + "\n")
        
        return True
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
