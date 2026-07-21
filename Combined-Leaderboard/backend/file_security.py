"""
Secure file handling for production deployment.
"""

import mimetypes
from pathlib import Path
from typing import Tuple, Optional
import logging
import uuid
import re
from werkzeug.utils import secure_filename

try:
    import magic  # python-magic; requires libmagic (not bundled on Windows)
    _MAGIC_AVAILABLE = True
except (ImportError, OSError):  # OSError raised when libmagic DLL is missing
    magic = None
    _MAGIC_AVAILABLE = False

from constants import (
    ALLOWED_FILE_EXTENSIONS,
    ALLOWED_MIME_TYPES,
    MAX_FILE_SIZE_PER_SUBMISSION,
    ERROR_INVALID_FILE_FORMAT,
    ERROR_FILE_TOO_LARGE,
)

logger = logging.getLogger(__name__)


class FileSecurityValidator:
    """Validates file uploads for security."""
    
    # Magic bytes for identifying file types
    MAGIC_BYTES = {
        b'\xff\xfe': 'UTF-16 LE',
        b'\xef\xbb\xbf': 'UTF-8 BOM',  # UTF-8 with BOM
        # We mostly expect text files
    }
    
    # Dangerous patterns in filenames
    DANGEROUS_PATTERNS = [
        r'\.\.[\\/]',  # Path traversal: ../
        r'[\x00-\x1f\x7f]',  # Control characters and log-breaking newlines
        r'[<>:"|?*]',   # Windows reserved chars
    ]
    
    @staticmethod
    def validate_filename(filename: str) -> Tuple[bool, Optional[str]]:
        """
        Validate filename for security issues.
        
        Args:
            filename: Original filename from upload
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not filename:
            return False, "Filename is empty"
        
        if len(filename) > 255:
            return False, "Filename too long (max 255 characters)"
        
        # Check for dangerous patterns
        for pattern in FileSecurityValidator.DANGEROUS_PATTERNS:
            if re.search(pattern, filename):
                logger.warning("Dangerous filename pattern detected: %r", filename)
                return False, "Filename contains invalid characters or patterns"
        
        # Check file extension
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_FILE_EXTENSIONS:
            return False, f"File extension {ext} not allowed"
        
        return True, None
    
    @staticmethod
    def validate_file_extension(filename: str) -> Tuple[bool, Optional[str]]:
        """Validate file extension."""
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_FILE_EXTENSIONS:
            logger.warning(f"Invalid file extension: {ext}")
            return False, ERROR_INVALID_FILE_FORMAT
        return True, None
    
    @staticmethod
    def validate_file_size(
        file_size: int,
        max_size: int = MAX_FILE_SIZE_PER_SUBMISSION
    ) -> Tuple[bool, Optional[str]]:
        """Validate file size."""
        if file_size <= 0:
            return False, "File is empty"
        
        if file_size > max_size:
            logger.warning(f"File exceeds size limit: {file_size} > {max_size}")
            if max_size == MAX_FILE_SIZE_PER_SUBMISSION:
                return False, ERROR_FILE_TOO_LARGE
            return False, f"File exceeds maximum size of {max_size} bytes"
        
        return True, None
    
    @staticmethod
    def validate_mime_type(file_content: bytes, filename: str) -> Tuple[bool, Optional[str]]:
        """
        Validate MIME type using python-magic.
        
        Args:
            file_content: File content (first 4096 bytes at least)
            filename: Original filename
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # python-magic may be unavailable (e.g. missing libmagic on Windows).
            # Fall back to extension-based validation in that case.
            if not _MAGIC_AVAILABLE:
                logger.debug(
                    "python-magic/libmagic unavailable; using extension-based "
                    "file validation instead of MIME sniffing."
                )
                return FileSecurityValidator.validate_file_extension(filename)

            # Try to detect MIME type from content
            mime = magic.Magic(mime=True)
            detected_mime = mime.from_buffer(file_content)
            
            # Check if detected MIME is allowed
            if detected_mime not in ALLOWED_MIME_TYPES:
                # Allow common text MIME types (some systems report text/* differently)
                if not detected_mime.startswith('text/'):
                    logger.warning(f"Invalid MIME type: {detected_mime}")
                    return False, ERROR_INVALID_FILE_FORMAT
            
            return True, None
            
        except Exception as e:
            logger.warning(f"Could not detect MIME type: {e}")
            # Fallback to extension check only
            return FileSecurityValidator.validate_file_extension(filename)
    
    @staticmethod
    def check_file_content(file_content: bytes, filename: str) -> Tuple[bool, Optional[str]]:
        """
        Check file content for malicious patterns.
        
        Args:
            file_content: File content
            filename: Original filename
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not file_content:
            return False, "File is empty"
        
        # Check for binary content that shouldn't be there
        # JSONL should be text.
        null_byte_count = file_content.count(b'\x00')
        if null_byte_count > 0:
            logger.warning(f"File contains null bytes: {filename}")
            return False, "File contains invalid binary content"
        
        # Check for extremely long lines (potential DOS)
        try:
            lines = file_content.split(b'\n')
            for line in lines[:100]:  # Check first 100 lines
                if len(line) > 100000:  # Max 100KB per line
                    logger.warning(f"Line too long in {filename}")
                    return False, "File contains unusually long lines"
        except Exception as e:
            logger.warning(f"Error checking file content: {e}")
        
        return True, None
    
    @staticmethod
    def generate_safe_filename(original_filename: str) -> str:
        """
        Generate a safe filename using UUID prefix.
        
        Args:
            original_filename: Original filename from upload
            
        Returns:
            Safe filename with UUID prefix
        """
        # Get extension
        ext = Path(original_filename).suffix.lower()
        
        # Generate unique name
        unique_id = str(uuid.uuid4())
        
        # Use secure_filename on the original name but replace with UUID
        safe_original = secure_filename(original_filename)
        
        # Combine UUID with safe original name
        safe_name = f"{unique_id}_{safe_original}"
        
        logger.info(f"Generated safe filename: {safe_name}")
        return safe_name
    
    @staticmethod
    def validate_and_secure_upload(
        file_stream,
        filename: str,
        max_size: int = MAX_FILE_SIZE_PER_SUBMISSION
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Comprehensive validation of uploaded file.
        
        Args:
            file_stream: File object from Flask request.files
            filename: Original filename
            max_size: Maximum allowed file size
            
        Returns:
            Tuple of (is_valid, error_message, safe_filename)
        """
        try:
            # 1. Validate filename
            is_valid, error = FileSecurityValidator.validate_filename(filename)
            if not is_valid:
                return False, error, None
            
            # 2. Validate file extension
            is_valid, error = FileSecurityValidator.validate_file_extension(filename)
            if not is_valid:
                return False, error, None
            
            # 3. Check file size
            file_stream.seek(0, 2)  # Seek to end
            file_size = file_stream.tell()
            file_stream.seek(0)  # Seek to beginning
            
            is_valid, error = FileSecurityValidator.validate_file_size(file_size, max_size=max_size)
            if not is_valid:
                return False, error, None
            
            # 4. Read content for validation
            file_content = file_stream.read(65536)  # Read first 64KB
            file_stream.seek(0)  # Reset for actual use
            
            if not file_content:
                return False, "File is empty", None
            
            # 5. Validate MIME type
            is_valid, error = FileSecurityValidator.validate_mime_type(file_content, filename)
            if not is_valid:
                return False, error, None
            
            # 6. Check file content
            is_valid, error = FileSecurityValidator.check_file_content(file_content, filename)
            if not is_valid:
                return False, error, None
            
            # 7. Generate safe filename
            safe_filename = FileSecurityValidator.generate_safe_filename(filename)
            
            logger.info(f"File validation passed: {filename} -> {safe_filename}")
            return True, None, safe_filename
            
        except Exception as e:
            logger.error(f"Unexpected error during file validation: {e}", exc_info=True)
            return False, "File validation error", None
