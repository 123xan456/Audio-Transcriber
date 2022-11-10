from pyannote.audio import Pipeline
from pydub import AudioSegment
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import textrank, re, os, whisper, pandas, time, json
from omegaconf import OmegaConf
from nemo.collections.asr.parts.utils.decoder_timestamps_utils import ASR_TIMESTAMPS
from nemo.collections.asr.parts.utils.diarization_utils import ASR_DIAR_OFFLINE
from pathlib import Path

UPLOAD_FOLDER = "uploads"
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

df = pandas.read_excel("/home/xiann/pyannote/results.xlsx", index_col=0)


@app.route("/")
def upload():
    return render_template("upload.html")


@app.route("/result", methods=["POST"])
def result():
    start = time.time()

    # obtain audio file, and save
    file = request.files["file"]
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
    audio = AudioSegment.from_wav(str(UPLOAD_FOLDER + "/" + filename))

    # load diarization and transciption models
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization@2.1",
        use_auth_token="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",
    )
    model = whisper.load_model("medium")

    # transcribe entire clip. split text every 500 words
    transcription = model.transcribe(str(UPLOAD_FOLDER + "/" + filename), language="en")
    transcription_list = transcription["text"].split()
    split_sections = [
        " ".join(transcription_list[i : i + 500])
        for i in range(0, len(transcription_list), 500)
    ]

    # summarize each section and join
    full_summary = ""
    for section in split_sections:
        summary = textrank.extract_sentences(section)
        full_summary += " " + summary

    # diarize
    AUDIO_FILENAME = str(UPLOAD_FOLDER + "/" + filename)
    CONFIG_NEMO = "config/diar_infer_telephonic.yaml"
    cfg_nemo = OmegaConf.load(CONFIG_NEMO)

    meta = {
        "audio_filepath": AUDIO_FILENAME,
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "num_speakers": 2,
        "rttm_filepath": None,
        "uem_filepath": None,
    }

    with open("input_manifest.json", "w") as fp:
        json.dump(meta, fp)
        fp.write("\n")

    cfg_nemo.diarizer.manifest_filepath = "input_manifest.json"
    pretrained_speaker_model = "titanet_large"
    cfg_nemo.diarizer.out_dir = (
        "../data"  # Directory to store intermediate files and prediction outputs
    )
    cfg_nemo.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
    cfg_nemo.diarizer.clustering.parameters.oracle_num_speakers = False

    # Using VAD generated from ASR timestamps
    cfg_nemo.diarizer.asr.model_path = "QuartzNet15x5Base-En"
    cfg_nemo.diarizer.oracle_vad = False  # ----> Not using oracle VAD
    cfg_nemo.diarizer.asr.parameters.asr_based_vad = True
    cfg_nemo.diarizer.asr.parameters.threshold = (
        100  # ASR based VAD threshold: If 100, all silences under 1 sec are ignored.
    )
    cfg_nemo.diarizer.asr.parameters.decoder_delay_in_sec = (
        0.2  # Decoder delay is compensated for 0.2 sec
    )
    asr_ts_decoder = ASR_TIMESTAMPS(**cfg_nemo.diarizer)
    asr_model = asr_ts_decoder.set_asr_model()
    word_hyp, word_ts_hyp = asr_ts_decoder.run_ASR(asr_model)

    asr_diar_offline = ASR_DIAR_OFFLINE(**cfg_nemo.diarizer)
    asr_diar_offline.word_ts_anchor_offset = asr_ts_decoder.word_ts_anchor_offset
    diar_hyp, diar_score = asr_diar_offline.run_diarization(cfg_nemo, word_ts_hyp)

    lines = []

    for li in diar_hyp[Path(str(UPLOAD_FOLDER + "/" + filename)).stem]:
        line_info = li.split()  # split string into [start, end, speaker]
        sp_start, sp_end, speaker = (
            int(float(line_info[0]) * 1000),
            int(float(line_info[1]) * 1000),
            line_info[2],
        )

        # add 2 second silence to beginning and end of section, cut audio by speaker, save, transcribe
        section = audio[sp_start:sp_end]
        silence = AudioSegment.silent(duration=2000)
        section = silence.append(section, crossfade=0)
        section = section.append(silence, crossfade=0)
        section.export("section.wav", format="wav")

        # transcribe
        line = model.transcribe("section.wav", language="en")
        lines.append([speaker[0], line["text"]])
        os.remove("section.wav")

    result = []
    for newline in lines:
        if newline[1] == "":
            continue
        result.append(str(newline[0]) + " : " + str(newline[1]))

    os.remove("uploads/" + filename)

    df.loc[len(df.index)] = [filename, result]
    df.to_excel("results.xlsx")

    time_taken = time.time() - start

    return render_template(
        "result.html",
        result=result,
        filename=filename,
        full_summary=full_summary,
        transcription=transcription["text"],
        time_taken=time_taken,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
