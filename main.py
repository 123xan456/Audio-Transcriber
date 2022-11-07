# @app.route("/result", methods=["POST"])
# def result():
#     pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token="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")
#     model = whisper.load_model("medium")
#     file = request.files["file"]
#     filename = secure_filename(file.filename)
#     file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

#     #add 1 second silence to beginning and end of uploaded audio file
#     audio = AudioSegment.from_wav(str(UPLOAD_FOLDER + "/" + filename))
#     silent  = AudioSegment.silent(duration=1000)
#     audio = silent.append(audio)
#     audio = audio.append(silent)
#     audio.export(str(UPLOAD_FOLDER + "/" + filename), format="wav")

#     dz = pipeline(str(UPLOAD_FOLDER + "/" + filename))

#     print(dz)

#     text_file = open("diarization.txt", "r+")
#     text_file.write(str(dz))
#     stringdz = str(dz).splitlines()
#     dzlist = []
#     lines = []
#     file_to_transcribe = silent

#     for l in stringdz:
#         # find start and end of each speaker section in milliseconds, and which speaker
#         start, end = tuple(re.findall("[0-9]+:[0-9]+:[0-9]+\.[0-9]+", string=l))
#         start = millisec(start)
#         end = millisec(end)
#         speaker = re.findall("SPEAKER_[0-9]+", string=l)
#         dzlist.append([start, end, speaker[0]])

#         # cut audio into speaker sections, append to file to be transcribed with silence in between

#         section = audio[start:end]
#         file_to_transcribe = file_to_transcribe + section + silent


#     file_to_transcribe.export("file_to_transcribe.wav", format="wav")
#     result = model.transcribe("file_to_transcribe.wav", language="en")
#     result = result["text"]

#     silent_parts = silence.detect_silence(file_to_transcribe)
#     print(silent_parts)
#     # os.remove("file_to_transcribe.wav")

#     # result = []

#     # for i in lines:
#     #     result.append(str(i[0]) + " : " + str(i[1]))
#     # print(result)

#     os.remove("uploads/" + filename)

#     df.loc[len(df.index)] = [filename, result]
#     df.to_excel("results.xlsx")
#     return render_template("result.html", result=result, filename=filename)

from pyannote.audio import Pipeline
from pydub import AudioSegment
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import textrank, math, re, os, whisper, pandas

UPLOAD_FOLDER = "uploads"
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

df = pandas.read_excel("results.xlsx", index_col=0)


def millisec(timeStr):
    spl = timeStr.split(":")
    millisec = (int)((int(spl[0]) * 3600 + int(spl[1]) * 60 + float(spl[2])) * 1000)
    return millisec


@app.route("/")
def upload():
    return render_template("upload.html")


@app.route("/result", methods=["POST"])
def result():
    # load diarization and transciption models
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization@2.1",
        use_auth_token="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",
    )
    model = whisper.load_model("medium")

    # obtain audio file, and save
    file = request.files["file"]
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

    # for summary of entire clip, 100 words every 10 mins
    start = 0
    end = 600000
    full_transcription = ""
    full_summary = ""
    dzlist = []
    lines = []
    print("hi3")

    # find number of 10min sections, pydub uses milliseconds
    audio = AudioSegment.from_wav(str(UPLOAD_FOLDER + "/" + filename))
    audio_length = audio.duration_seconds * 1000
    num_sections = math.ceil(audio_length / 600000)
    print("hi4")

    for i in range(num_sections):
        section = audio[start:end]
        print("reach")
        # add 2 second silence to beginning and end of section
        silence = AudioSegment.silent(duration=2000)
        section = silence.append(section, crossfade=0)
        section = section.append(silence, crossfade=0)
        section.export("section.wav", format="wav")

        meeting_in_text = model.transcribe("section.wav", language="en")
        summary = textrank.extract_sentences(meeting_in_text["text"])
        print(summary)
        full_transcription += meeting_in_text["text"]
        full_summary += summary

        dz = pipeline("section.wav")
        stringdz = str(dz).splitlines()

        for l in stringdz:
            # find start and end of each speaker section in milliseconds, and which speaker
            sp_start, sp_end = tuple(
                re.findall("[0-9]+:[0-9]+:[0-9]+\.[0-9]+", string=l)
            )
            sp_start = millisec(sp_start)
            sp_end = millisec(sp_end)

            speaker = re.findall("SPEAKER_[0-9]+", string=l)
            dzlist.append([sp_start, sp_end, speaker[0]])

            # cut audio into speaker sections, save each section, transcribe, then delete
            section = audio[sp_start:sp_end]
            section.export("section.wav", format="wav")
            line = model.transcribe("section.wav", language="en")
            lines.append([speaker[0], line["text"]])
            os.remove("section.wav")

        start += 600000
        end += 600000

    result = []
    for line in lines:
        if line[1] == "":
            continue
        result.append(str(line[0]) + " : " + str(line[1]))

    os.remove("uploads/" + filename)

    df.loc[len(df.index)] = [filename, result]
    df.to_excel("results.xlsx")

    return render_template(
        "result.html",
        result=result,
        filename=filename,
        full_summary=full_summary,
        full_transcription=full_transcription,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
