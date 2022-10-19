# import whisper

# model = whisper.load_model("small")

# result = model.transcribe("shortened.wav", language="en")
# print(result["text"])

import pandas as pd

df = pd.read_csv("results.csv")
print(df)
