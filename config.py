import os

from dotenv import load_dotenv

load_dotenv()

BEAMNG_HOME = os.getenv(
    "BEAMNG_HOME", r"C:\Program Files (x86)\Steam\steamapps\common\BeamNG.drive"
)
BEAMNG_USER = os.getenv("BEAMNG_USER", None)
HEADLESS = os.getenv("HEADLESS", "false").lower() == "true"

# Lua console log toggles
LOG_LIDAR = os.getenv("LOG_LIDAR", "true").lower() == "true"
LOG_RADAR = os.getenv("LOG_RADAR", "true").lower() == "true"
LOG_CAMERA = os.getenv("LOG_CAMERA", "true").lower() == "true"
LOG_CHECKPOINT_HIT = os.getenv("LOG_CHECKPOINT_HIT", "true").lower() == "true"
LOG_CHECKPOINT_RESPAWN = os.getenv("LOG_CHECKPOINT_RESPAWN", "true").lower() == "true"
LOG_CHECKPOINT_WARN = os.getenv("LOG_CHECKPOINT_WARN", "true").lower() == "true"
