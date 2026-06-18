"""
Database initialization and migration utilities.
Migrates from JSON file-based storage to SQLite/PostgreSQL.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import sqlalchemy as sa
from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

Base = declarative_base()


class Submission(Base):
    """Database model for submission records."""
    __tablename__ = "submissions"

    submission_id = Column(String(36), primary_key=True, index=True)
    model_name = Column(String(255), index=True, nullable=False)
    benchmark = Column(String(50), index=True, nullable=False)
    overall_accuracy = Column(Float, nullable=False)
    total_samples = Column(Integer, nullable=False)
    correct_samples = Column(Integer, nullable=False)
    submitted_at = Column(DateTime, index=True, nullable=False)
    task_accuracy = Column(JSON, nullable=True)  # Store task-level results
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Submission {self.submission_id} - {self.model_name} ({self.overall_accuracy:.2%})>"


class Database:
    """Database connection and migration manager."""

    def __init__(self, database_url: str = "sqlite:///leaderboard.db"):
        """
        Initialize database connection.
        
        Args:
            database_url: Database connection string
        """
        self.database_url = database_url
        self.engine = create_engine(
            database_url,
            echo=False,
            pool_size=10,
            max_overflow=20,
            pool_recycle=3600,
            connect_args={"check_same_thread": False} if database_url.startswith("sqlite") else {}
        )
        self.Session = sessionmaker(bind=self.engine)
        logger.info(f"Database initialized: {database_url}")

    def create_tables(self):
        """Create all tables."""
        try:
            Base.metadata.create_all(self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating tables: {e}", exc_info=True)
            raise

    def migrate_from_json(self, json_file: Path):
        """
        Migrate submissions from JSON file to database.
        
        Args:
            json_file: Path to leaderboard.json file
        """
        if not json_file.exists():
            logger.warning(f"JSON file not found: {json_file}")
            return

        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            session = self.Session()
            migrated_count = 0

            try:
                for benchmark, submissions in data.items():
                    if not isinstance(submissions, list):
                        continue

                    for submission_data in submissions:
                        try:
                            # Check if already exists
                            existing = session.query(Submission).filter_by(
                                submission_id=submission_data["submission_id"]
                            ).first()

                            if existing:
                                logger.debug(f"Skipping existing submission: {submission_data['submission_id']}")
                                continue

                            # Create new record
                            submission = Submission(
                                submission_id=submission_data["submission_id"],
                                model_name=submission_data["model_name"],
                                benchmark=benchmark,
                                overall_accuracy=submission_data["overall_accuracy"],
                                total_samples=submission_data["total_samples"],
                                correct_samples=submission_data["correct_samples"],
                                submitted_at=datetime.fromisoformat(submission_data["submitted_at"]),
                                task_accuracy=submission_data.get("task_accuracy", {}),
                                metadata={},
                            )

                            session.add(submission)
                            migrated_count += 1

                        except Exception as e:
                            logger.error(f"Error migrating submission: {e}")
                            continue

                session.commit()
                logger.info(f"Successfully migrated {migrated_count} submissions")

            finally:
                session.close()

        except Exception as e:
            logger.error(f"Migration failed: {e}", exc_info=True)
            raise

    def get_submission(self, submission_id: str) -> Optional[dict]:
        """Get submission by ID."""
        session = self.Session()
        try:
            submission = session.query(Submission).filter_by(
                submission_id=submission_id
            ).first()

            if not submission:
                return None

            return {
                "submission_id": submission.submission_id,
                "model_name": submission.model_name,
                "benchmark": submission.benchmark,
                "overall_accuracy": submission.overall_accuracy,
                "total_samples": submission.total_samples,
                "correct_samples": submission.correct_samples,
                "submitted_at": submission.submitted_at.isoformat(),
                "task_results": submission.task_accuracy or {},
                "metadata": submission.metadata or {},
            }
        finally:
            session.close()

    def get_leaderboard(
        self,
        benchmark: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> list:
        """Get leaderboard entries."""
        session = self.Session()
        try:
            query = session.query(Submission)

            if benchmark:
                query = query.filter_by(benchmark=benchmark)

            query = query.order_by(Submission.overall_accuracy.desc())
            submissions = query.limit(limit).offset(offset).all()

            results = []
            for rank, submission in enumerate(submissions, 1):
                results.append({
                    "rank": rank,
                    "submission_id": submission.submission_id,
                    "model_name": submission.model_name,
                    "benchmark": submission.benchmark,
                    "overall_accuracy": submission.overall_accuracy,
                    "total_samples": submission.total_samples,
                    "correct_samples": submission.correct_samples,
                    "submitted_at": submission.submitted_at.isoformat(),
                    "task_accuracy": submission.task_accuracy or {},
                })

            return results
        finally:
            session.close()

    def add_submission(self, submission_dict: dict) -> bool:
        """Add new submission to database."""
        session = self.Session()
        try:
            submission = Submission(
                submission_id=submission_dict["submission_id"],
                model_name=submission_dict["model_name"],
                benchmark=submission_dict["benchmark"],
                overall_accuracy=submission_dict["overall_accuracy"],
                total_samples=submission_dict["total_samples"],
                correct_samples=submission_dict["correct_samples"],
                submitted_at=submission_dict["submitted_at"],
                task_accuracy=submission_dict.get("task_results", {}),
                metadata=submission_dict.get("metadata", {}),
            )

            session.add(submission)
            session.commit()

            logger.info(f"Added submission: {submission.submission_id}")
            return True

        except Exception as e:
            logger.error(f"Error adding submission: {e}", exc_info=True)
            session.rollback()
            return False
        finally:
            session.close()

    def get_statistics(self, benchmark: Optional[str] = None) -> dict:
        """Get leaderboard statistics."""
        session = self.Session()
        try:
            query = session.query(Submission)

            if benchmark:
                query = query.filter_by(benchmark=benchmark)

            total_submissions = query.count()

            if total_submissions == 0:
                return {
                    "total_submissions": 0,
                    "unique_models": 0,
                    "average_accuracy": 0.0,
                    "best_accuracy": 0.0,
                }

            unique_models = session.query(Submission.model_name.distinct()).count()
            
            # Calculate average accuracy weighted by total_samples
            submissions = query.all()
            total_correct = sum(s.correct_samples for s in submissions)
            total_samples = sum(s.total_samples for s in submissions)
            
            average_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
            best_accuracy = max(s.overall_accuracy for s in submissions) if submissions else 0.0

            return {
                "total_submissions": total_submissions,
                "unique_models": unique_models,
                "average_accuracy": average_accuracy,
                "best_accuracy": best_accuracy,
            }

        finally:
            session.close()


if __name__ == "__main__":
    # Example: Initialize database and migrate from JSON
    import os
    from logging_config import logger as setup_logger

    # Setup logging
    setup_logger

    # Initialize database
    db_url = os.getenv("DATABASE_URL", "sqlite:///leaderboard.db")
    db = Database(db_url)

    # Create tables
    db.create_tables()

    # Migrate from JSON if it exists
    json_file = Path("results/leaderboard.json")
    if json_file.exists():
        print(f"Migrating from {json_file}...")
        db.migrate_from_json(json_file)
    else:
        print(f"No JSON file found at {json_file}")

    print("Database initialization complete!")
