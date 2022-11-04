import whisper
import os
import pandas as pd
from pyannote.audio import Pipeline
from pydub import AudioSegment
import re
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from pathlib import Path
import textrank

rootpath = Path(__file__).parent
sectionpath = rootpath/"section.wav"

UPLOAD_FOLDER = "uploads"
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

df = pd.read_excel("results.xlsx", index_col=0)


def millisec(timeStr):
    spl = timeStr.split(":")
    s = (int)((int(spl[0]) * 3600 + int(spl[1]) * 60 + float(spl[2])) * 1000)
    return s      


@app.route("/")
def upload():
    print("reached!")
    return render_template("upload.html")


@app.route("/result", methods=["POST"])
def result():
    print("reached!")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token="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")
    model = whisper.load_model("large")
    file = request.files["file"]
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

    # for summary of entire clip, 100 words
    meeting_in_text = model.transcribe(str(UPLOAD_FOLDER + "/" + filename), language = "en")
    summary = textrank.extract_sentences(meeting_in_text["text"])

    #add 2 second silence to beginning and end of uploaded audio file
    audio = AudioSegment.from_wav(str(UPLOAD_FOLDER + "/" + filename))
    silence  = AudioSegment.silent(duration=2000)
    audio = silence.append(audio, crossfade=0)
    audio = audio.append(silence, crossfade=0)
    audio.export(str(UPLOAD_FOLDER + "/" + filename), format="wav")

    dz = pipeline(str(UPLOAD_FOLDER + "/" + filename))

    print(dz)

    text_file = open("diarization.txt", "r+")
    text_file.write(str(dz))
    stringdz = str(dz).splitlines()

    dzlist = []
    lines = []

    for l in stringdz:
        # find start and end of each speaker section in milliseconds, and which speaker
        start, end = tuple(re.findall("[0-9]+:[0-9]+:[0-9]+\.[0-9]+", string=l))
        start = millisec(start)
        end = millisec(end)

        if end - start < 1000:
            continue

        speaker = re.findall("SPEAKER_[0-9]+", string=l)
        dzlist.append([start, end, speaker[0]])


        # cut audio into speaker sections, save each section, transcribe, then delete
        section = audio[start:end]
        section.export(str(sectionpath), format="wav")
        line = model.transcribe(str(sectionpath), language="en")
        lines.append([speaker[0], line["text"]])
        os.remove(str(sectionpath))

    result = []

    for i in lines:
        result.append(str(i[0]) + " : " + str(i[1]))
    print(result)
    os.remove("uploads/" + filename)

    df.loc[len(df.index)] = [filename, result]
    df.to_excel("results.xlsx")
    return render_template("result.html", result=result, filename=filename, summary = summary)


if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 8000)