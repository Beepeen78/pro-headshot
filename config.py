"""
Configuration settings for the Pro Headshot Generator application.
"""
import os
from pathlib import Path

# Device configuration will be set in app.py after torch import

# File paths
TEMP_DIR = Path(os.getenv("TEMP_DIR", "temp_downloads"))
TEMP_DIR.mkdir(exist_ok=True, parents=True)

# File cleanup settings
MAX_FILE_AGE_HOURS = int(os.getenv("MAX_FILE_AGE_HOURS", "24"))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))

# Image validation settings
MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", "10"))
ALLOWED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".webp"}
MIN_IMAGE_DIMENSION = 128
MAX_IMAGE_DIMENSION = 4096

# Generation settings
DEFAULT_NUM_STEPS = int(os.getenv("DEFAULT_NUM_STEPS", "30"))
DEFAULT_GUIDANCE_SCALE = float(os.getenv("DEFAULT_GUIDANCE_SCALE", "5.0"))
DEFAULT_SEED = int(os.getenv("DEFAULT_SEED", "42"))

# Prompt validation
MAX_PROMPT_LENGTH = int(os.getenv("MAX_PROMPT_LENGTH", "500"))
MAX_NEGATIVE_PROMPT_LENGTH = int(os.getenv("MAX_NEGATIVE_PROMPT_LENGTH", "500"))

