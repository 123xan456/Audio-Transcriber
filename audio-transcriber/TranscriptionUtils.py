import whisper

from Configs import RESOURCES


class Transcriber:
    def __init__(self) -> None:
        # Model names: tiny base small medium large-v1 large-v2
        self.model = whisper.load_model(
            "large-v2", device="cuda", download_root=RESOURCES / "models" 
        )

    def transcribe(self, audioPath, lang="en"):
        transcription = self.model.transcribe(
            audioPath, language=lang, condition_on_previous_text=False
        )
        return transcription

