# config.py
import os
from dotenv import load_dotenv

load_dotenv()

def get_env_int(var_name, default):
    try:
        return int(os.getenv(var_name, default))
    except ValueError:
        print(f"Warning: {var_name} is not a valid int, using default {default}")
        return default

def get_env_float(var_name, default):
    try:
        return float(os.getenv(var_name, default))
    except ValueError:
        print(f"Warning: {var_name} is not a valid float, using default {default}")
        return default


FPS = get_env_int("FPS", 30)
WINDOW_SIZE = get_env_int("WINDOW_SIZE", 30)
V_THRESH = get_env_int("V_THRESH", 0)
DY_THRESH = get_env_int("DY_THRESH", 0)
ASPECT_RATIO_THRESH = get_env_float("ASPECT_RATIO_THRESH", 0)
