"""
Application constants and configuration limits.
"""

import os

# File upload limits
MAX_FILE_SIZE = int(os.getenv("MAX_CONTENT_LENGTH", str(50 * 1024 * 1024)))
MAX_FILE_SIZE_PER_SUBMISSION = int(
    os.getenv("MAX_FILE_SIZE_PER_SUBMISSION", str(MAX_FILE_SIZE))
)
MAX_SPATIAL_ARCHIVE_BYTES = int(
    os.getenv("MAX_SPATIAL_ARCHIVE_BYTES", str(10 * 1024 * 1024))
)
MAX_SPATIAL_MULTIPART_BYTES = int(
    os.getenv(
        "MAX_SPATIAL_MULTIPART_BYTES",
        str(MAX_SPATIAL_ARCHIVE_BYTES + 2 * 1024 * 1024),
    )
)
MAX_SPATIAL_SUBMISSION_BYTES = int(
    os.getenv("MAX_SPATIAL_SUBMISSION_BYTES", str(20 * 1024 * 1024))
)
MAX_SPATIAL_MANIFEST_BYTES = int(
    os.getenv("MAX_SPATIAL_MANIFEST_BYTES", str(1024 * 1024))
)
MAX_SPATIAL_REPORT_BYTES = int(
    os.getenv("MAX_SPATIAL_REPORT_BYTES", str(2 * 1024 * 1024))
)
MAX_SPATIAL_ZIP_COMPRESSION_RATIO = int(
    os.getenv("MAX_SPATIAL_ZIP_COMPRESSION_RATIO", "100")
)
ALLOWED_FILE_EXTENSIONS = {'.jsonl'}
ALLOWED_MIME_TYPES = {'application/jsonl', 'application/x-ndjson', 'application/json', 'text/plain'}

# Model name constraints
MAX_MODEL_NAME_LENGTH = 255
MIN_MODEL_NAME_LENGTH = 1

# API constraints
DEFAULT_LEADERBOARD_LIMIT = 25
MAX_LEADERBOARD_LIMIT = 1000
MIN_LEADERBOARD_LIMIT = 1

# Rate limiting
SUBMISSIONS_PER_HOUR = 3
SUBMISSIONS_PER_DAY = 10
# Authoritative DB-backed per-account, per-benchmark submission quota over a
# rolling 24-hour window. Each benchmark has an independent allowance.
SUBMISSION_DAILY_LIMIT_PER_BENCHMARK = int(
    os.getenv("SUBMISSION_DAILY_LIMIT_PER_BENCHMARK", "1")
)
if SUBMISSION_DAILY_LIMIT_PER_BENCHMARK <= 0:
    raise RuntimeError(
        "SUBMISSION_DAILY_LIMIT_PER_BENCHMARK must be a positive whole number."
    )
# Compatibility alias for older imports. The value is now per benchmark.
SUBMISSION_DAILY_LIMIT = SUBMISSION_DAILY_LIMIT_PER_BENCHMARK
SUBMISSION_RESERVATION_TIMEOUT_MINUTES = int(
    os.getenv("SUBMISSION_RESERVATION_TIMEOUT_MINUTES", "15")
)
if SUBMISSION_RESERVATION_TIMEOUT_MINUTES <= 0:
    raise RuntimeError(
        "SUBMISSION_RESERVATION_TIMEOUT_MINUTES must be a positive whole number."
    )
LEADERBOARD_REQUESTS_PER_MINUTE = 60
API_REQUESTS_PER_MINUTE = 100

# Prediction constraints
MIN_PREDICTIONS_REQUIRED = 1
MAX_PREDICTIONS_PER_SUBMISSION = 1000000
MAX_SUBMISSION_LINE_CHARS = int(
    os.getenv("MAX_SUBMISSION_LINE_CHARS", "100000")
)
if MAX_SUBMISSION_LINE_CHARS <= 0:
    raise RuntimeError(
        "MAX_SUBMISSION_LINE_CHARS must be a positive whole number."
    )

# Answer extraction
VALID_OPTIONS = {'A', 'B', 'C', 'D', 'E', 'F'}
OPTION_PATTERN_PRIORITY = [
    'parentheses',      # (A), (B), etc.
    'word_prefix',      # "Option A", "Choice B"
    'standalone',       # Just "A" or "B"
    'first_char',       # First character if valid option
]

# Statistical constants
MIN_SAMPLES_FOR_STDEV = 2
ACCURACY_DECIMAL_PLACES = 4
CONFIDENCE_INTERVAL = 0.95  # 95% CI

# Logging
LOG_DIR = "logs"
LOG_LEVEL = "INFO"
MAX_LOG_FILE_SIZE = 10 * 1024 * 1024  # 10MB
BACKUP_LOG_COUNT = 10

# Database
DEFAULT_DB_URL = "sqlite:///leaderboard.db"
CONNECTION_POOL_SIZE = 10
MAX_OVERFLOW = 20
POOL_RECYCLE = 3600  # Recycle connections after 1 hour

# Caching
CACHE_TTL_SECONDS = 300  # 5 minutes
GT_CACHE_TTL_SECONDS = 3600  # 1 hour

# Timeouts
FILE_PROCESSING_TIMEOUT = 300  # 5 minutes
GT_LOADING_TIMEOUT = 60  # 1 minute
API_TIMEOUT = 30  # 30 seconds

# Error messages
ERROR_INVALID_BENCHMARK = "Invalid benchmark. Must be 'minds_eye', 'do_you_see_me', or 'spatial'"
ERROR_INVALID_FILE_FORMAT = "Only JSONL files are supported"
ERROR_FILE_TOO_LARGE = f"File exceeds maximum size of {MAX_FILE_SIZE_PER_SUBMISSION / 1024 / 1024}MB"
ERROR_INVALID_MODEL_NAME = f"Model name must be 1-{MAX_MODEL_NAME_LENGTH} characters, alphanumeric with hyphens/underscores"
ERROR_NO_PREDICTIONS_FOUND = "No valid predictions found in submission"
ERROR_MALFORMED_CSV = "CSV submissions are no longer supported"
ERROR_MALFORMED_JSON = "JSONL file is malformed or missing required fields"
