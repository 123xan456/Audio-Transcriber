from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
dz = pipeline(str("audio/Sample 3.wav"))

for turn, _, speaker in dz.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s {speaker}")