from beamng_env import BeamNGDrivingEnv
from config import BEAMNG_HOME, BEAMNG_USER

if __name__ == "__main__":
    env = BeamNGDrivingEnv(
        beamng_home=BEAMNG_HOME,
        beamng_user=BEAMNG_USER,
    )

    env.human_play()

    input("Press Enter to quit...")
    env.close()
