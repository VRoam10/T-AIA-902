import os

from dotenv import load_dotenv

load_dotenv()

BEAMNG_HOME = os.getenv(
    "BEAMNG_HOME", r"C:\Program Files (x86)\Steam\steamapps\common\BeamNG.drive"
)
BEAMNG_USER = os.getenv("BEAMNG_USER", None)
