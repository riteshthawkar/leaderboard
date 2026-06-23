"""
Simple user authentication database.
Users register with username + password; on success they receive an API token
that they use as a Bearer token for submission. Rate limits are keyed on the
token (and therefore on the user account).
"""

import secrets
from datetime import datetime
from pathlib import Path
from typing import Optional

import sqlalchemy as sa
from sqlalchemy import create_engine, Column, String, DateTime, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from werkzeug.security import generate_password_hash, check_password_hash

Base = declarative_base()

# Constant-time sentinel: always run check_password_hash even for unknown users
# to prevent timing-based username enumeration.
_DUMMY_HASH: str = generate_password_hash("__dummy_sentinel__")

# DB file lives next to the leaderboard store
_DB_PATH = Path(__file__).resolve().parent.parent / "users.db"
_DB_URL = f"sqlite:///{_DB_PATH}"

_engine = create_engine(
    _DB_URL,
    echo=False,
    connect_args={"check_same_thread": False},
)
_Session = sessionmaker(bind=_engine)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(80), unique=True, nullable=False, index=True)
    password_hash = Column(String(256), nullable=False)
    api_token = Column(String(64), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<User {self.username}>"


def init_db() -> None:
    """Create tables if they don't exist."""
    Base.metadata.create_all(_engine)


def register_user(username: str, password: str) -> Optional[str]:
    """
    Create a new user. Returns the API token on success, or None if the
    username is already taken.
    """
    username = username.strip().lower()
    if not username or not password:
        return None
    with _Session() as session:
        if session.query(User).filter_by(username=username).first():
            return None  # username already taken
        token = secrets.token_urlsafe(32)
        user = User(
            username=username,
            password_hash=generate_password_hash(password),
            api_token=token,
        )
        session.add(user)
        session.commit()
        return token


def login_user(username: str, password: str) -> Optional[str]:
    """
    Verify credentials. Returns the user's API token on success, else None.
    Always runs check_password_hash regardless of whether the user exists to
    prevent timing-based username enumeration.
    """
    username = username.strip().lower()
    with _Session() as session:
        user = session.query(User).filter_by(username=username).first()
        target_hash = user.password_hash if user else _DUMMY_HASH
        match = check_password_hash(target_hash, password)
        if match and user:
            return user.api_token
        return None


def get_username_by_token(token: str) -> Optional[str]:
    """Return the username for a given API token, or None if invalid."""
    with _Session() as session:
        user = session.query(User).filter_by(api_token=token).first()
        return user.username if user else None
