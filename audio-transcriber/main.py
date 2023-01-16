import os
import pandas as pd
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from Configs import RESOURCES
from DiarizationUtils import Diarizer
from SummarizationUtils import Summarizer
from TranscriptionUtils import Transcriber

transcriber = Transcriber()
diarizer = Diarizer()
summarizer = Summarizer()

app = Flask(__name__, template_folder=str(RESOURCES/"templates"))
app.config["UPLOAD_FOLDER"] = str(RESOURCES/"uploads")

df = pd.read_excel(str(RESOURCES/"results.xlsx"), index_col=0)

print("startup complete")

def millisec(timeStr):
    spl = timeStr.split(":")
    s = (int)((int(spl[0]) * 3600 + int(spl[1]) * 60 + float(spl[2])) * 1000)
    return s

@app.route("/")
def upload():
    return render_template("upload.html")


@app.route("/result", methods=["POST"])
def result():
    
    file = request.files["file"]
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
    filePath = str(RESOURCES/"uploads"/filename)

    resultInitial = transcriber.transcribe(filePath)
    
    diarized = diarizer.diarize(resultInitial, filePath)
    
    result = summarizer.summarize(diarized, maxLen=400, minLen=100, lengthPenalty=2.0, repetitionPenalty=1.2)

    os.remove("uploads/" + filename)

    df.loc[len(df.index)] = [filename, result]
    df.to_excel("results.xlsx")
    return render_template("result.html", result=result, filename=filename)


