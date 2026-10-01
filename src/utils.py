import os
from pathlib import Path

# Set the project root directory (moves up one level from the 'src' folder)
BASE_DIR = Path(__file__).resolve().parent.parent

# Define absolute paths for data and model directories relative to the root
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Automatically create the 'models' directory if it doesn't exist to prevent IO errors
os.makedirs(MODELS_DIR, exist_ok=True)

def get_data_path(filename):
    """
    Constructs the absolute path for a file inside the 'data' directory.
    Ensures the script can locate datasets regardless of the execution context.
    """
    return str(DATA_DIR / filename)

def get_model_path(filename):
    """
    Constructs the absolute path for a file inside the 'models' directory.
    Used for serializing (saving) or deserializing (loading) the .pkl model artifacts.
    """
    return str(MODELS_DIR / filename)