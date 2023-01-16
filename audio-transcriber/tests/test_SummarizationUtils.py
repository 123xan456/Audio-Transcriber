import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
RESOURCES = Path(__file__).parent/"res"

from SummarizationUtils import Summarizer
from TranscriptionUtils import Transcriber

class Test_SummarizationUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.summarizer = Summarizer()
        self.transcriber = Transcriber()

    def tearDown(self) -> None:
        pass

    def testSummarize(self):
        audioPath = str(RESOURCES/"input"/"louder.wav")
        transcription = self.transcriber.transcribe(audioPath, lang="en")
        results =  self.summarizer.summarize(transcription["text"])
        print(results)

    
    
    
    
    