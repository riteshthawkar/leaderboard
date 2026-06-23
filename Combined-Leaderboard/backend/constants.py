"""
Application constants and configuration limits.
"""

# File upload limits
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_FILE_SIZE_PER_SUBMISSION = 10 * 1024 * 1024  # 10MB per submission
ALLOWED_FILE_EXTENSIONS = {'.csv', '.json'}
ALLOWED_MIME_TYPES = {'text/csv', 'application/json', 'text/plain'}

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
LEADERBOARD_REQUESTS_PER_MINUTE = 60
API_REQUESTS_PER_MINUTE = 100

# Prediction constraints
MIN_PREDICTIONS_REQUIRED = 1
MAX_PREDICTIONS_PER_SUBMISSION = 1000000

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
ERROR_INVALID_BENCHMARK = "Invalid benchmark. Must be 'minds_eye' or 'do_you_see_me'"
ERROR_INVALID_FILE_FORMAT = "Only CSV and JSON files are supported"
ERROR_FILE_TOO_LARGE = f"File exceeds maximum size of {MAX_FILE_SIZE_PER_SUBMISSION / 1024 / 1024}MB"
ERROR_INVALID_MODEL_NAME = f"Model name must be 1-{MAX_MODEL_NAME_LENGTH} characters, alphanumeric with hyphens/underscores"
ERROR_NO_PREDICTIONS_FOUND = "No valid predictions found in submission"
ERROR_MALFORMED_CSV = "CSV file is malformed or missing required columns"
ERROR_MALFORMED_JSON = "JSON file is malformed or missing required fields"
