import os
import pandas as pd
from pathlib import Path
from flask import Flask, render_template, request
from pydub import AudioSegment
from werkzeug.utils import secure_filename

from Configs import RESOURCES
from DiarizationUtils import Diarizer
from LoggingUtils import MainLogger
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
    MainLogger.logger.info("Reached home page")
    return render_template("upload.html")


@app.route("/result", methods=["POST"])
def result():
    MainLogger.logger.info("Started processing audio clip")
    file = request.files["file"]
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
    inputPath = str(RESOURCES/"uploads"/filename)
    outputPath = str(RESOURCES/"uploads"/"converted.wav")

    # Converting .mp3 to .wav
    extension = Path(inputPath).suffix
    MainLogger.logger.info(f"Input file type: {extension}")
    if extension == ".mp3":
        MainLogger.logger.info("Converting audio clip from .mp3 to .wav")
        AudioSegment.from_mp3(inputPath).export(outputPath, format="wav")
    else:
        outputPath = inputPath

    # Converting audio with more than 1 channel to 1 channel
    MainLogger.logger.info("Converting audio clip from multi-channel to single-channel")
    sound = AudioSegment.from_wav(outputPath)
    sound = sound.set_channels(1)
    
    sound.export(outputPath, format="wav")

    MainLogger.logger.info("Started transcribing audio clip")
    resultInitial = transcriber.transcribe(outputPath)
    
    MainLogger.logger.info("Started diarizing audio clip")
    diarized = diarizer.diarize(resultInitial, outputPath)
    
    MainLogger.logger.info("Started summarizing audio clip")
    result = summarizer.summarize(diarized, maxLen=400, minLen=100, lengthPenalty=2.0, repetitionPenalty=1.2)

    os.remove(inputPath)

    df.loc[len(df.index)] = [filename, result]
    df.to_excel(str(RESOURCES/"results.xlsx"))
    return render_template("result.html", result=result, filename=filename)


