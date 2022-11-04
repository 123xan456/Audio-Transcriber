import whisper

model = whisper.load_model("small")

result = model.transcribe("audio/Sample 3.wav",language = "en")
print(result["text"])