import sys
import unittest
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
RESOURCES = Path(__file__).parent/"res"

from TranscriptionUtils import Transcriber

class Test_TranscriptionUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.transcriber = Transcriber()

    def tearDown(self) -> None:
        pass

    def testTranscribe(self):
        audioPath = str(RESOURCES/"input"/"louder.wav")

        transcription = self.transcriber.transcribe(audioPath, lang="en")
        print(transcription["text"] + "\n")
        df = pd.DataFrame(transcription["segments"])
        df = df.set_index("id")
        df.to_csv(str(RESOURCES/"output"/"transcription.csv"))