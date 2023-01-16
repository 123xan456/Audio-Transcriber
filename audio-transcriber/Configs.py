import nltk
import os
from pathlib import Path
from filelock import FileLock
from torch import hub

from LoggingUtils import MainLogger

# Startup Configurations
portNumber = 5000
processWorkers = 1

# Paths
ROOT = Path(__file__).parent
RESOURCES = ROOT/"res"
TEMPLATES = RESOURCES/"templates"
CACHE_DIRECTORY = ROOT/'.cache'

# Setting a few environment variables
os.environ["TRANSFORMERS_CACHE"] = str(Path(os.getcwd()) / ".cache")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ['NEMO_CACHE_DIR'] = str(CACHE_DIRECTORY)
nltk.data.path.append(RESOURCES)
hub.set_dir(str(CACHE_DIRECTORY))

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    with FileLock(".lock") as lock:
        nltk.download("punkt", download_dir=RESOURCES, quiet=False)

MainLogger.logger.info("Successfully loaded all configurations")