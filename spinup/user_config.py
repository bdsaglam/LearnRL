# Where experiment outputs are saved by default:
import pathlib

DEFAULT_DATA_DIR = pathlib.Path.home() / "spinup-data"

# Whether to automatically insert a date and time stamp into the names of
# save directories:
FORCE_DATESTAMP = False

# Whether GridSearch provides automatically-generated default shorthands:
DEFAULT_SHORTHAND = True

# Tells the GridSearch how many seconds to pause for before launching 
# experiments.
WAIT_BEFORE_LAUNCH = 3
