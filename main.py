import whisper
import os
import pandas as pd
from pyannote.audio import Pipeline
from pydub import AudioSegment
import re
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from pathlib import Path

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
    return render_template("upload.html")


@app.route("/result", methods=["POST"])
def result():
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
    model = whisper.load_model("small")
    file = request.files["file"]
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

    audio = AudioSegment.from_wav(str(UPLOAD_FOLDER + "/" + filename))
    dz = pipeline(str(UPLOAD_FOLDER + "/" + filename))
    text_file = open("diarization.txt", "r+")
    text_file.write(str(dz))
    stringdz = str(dz).splitlines()

    dzlist = []
    lines = []

    for l in stringdz:
        # find start and end of each speaker section in milliseconds, and which speaker
        start, end = tuple(re.findall("[0-9]+:[0-9]+:[0-9]+\.[0-9]+", string=l))
        start = millisec(start) - 500
        end = millisec(end) + 500
        speaker = re.findall("SPEAKER_[0-9]+", string=l)
        dzlist.append([start, end, speaker[0]])

        # cut audio into speaker sections, save each section, transcribe, then delete
        section = audio[start:end]
        section.export(str(sectionpath), format="wav")
        line = model.transcribe(str(sectionpath), language="en")
        lines.append([start, end, speaker[0], line["text"]])
        os.remove(str(sectionpath))

    result = ""

    for i in lines:
        result = result + str(i[2]) + " : " + str(i[3]) + "\n"
    print(result)
    os.remove("uploads/" + filename)

    df.loc[len(df.index)] = [filename, result]
    df.to_excel("results.xlsx")
    return render_template("result.html", result=result, filename=filename)


if __name__ == "__main__":
    app.run(debug=True)
