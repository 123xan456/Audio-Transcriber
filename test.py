# import os
# from pyannote.audio import Pipeline
# from pydub import AudioSegment
# import re
# import whisper
# from pathlib import Path

# rootpath = Path(__file__).parent
# sectionpath = rootpath/"section.wav"

# pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
# model = whisper.load_model("small")
# audiopath = "audio/shortened.wav"

# # only for wav files, pyannote only suports wav files
# audio = AudioSegment.from_wav(audiopath)
# dz = pipeline(audiopath)
# dzlist = []
# lines = []

# text_file = open("diarization.txt", "r+")
# text_file.write(str(dz))
# stringdz = str(dz).splitlines()


# def millisec(timeStr):
#     spl = timeStr.split(":")
#     s = (int)((int(spl[0]) * 3600 + int(spl[1]) * 60 + float(spl[2])) * 1000)
#     return s


# for l in stringdz:
#     # find start and end of each speaker section in milliseconds, and which speaker
#     start, end = tuple(re.findall("[0-9]+:[0-9]+:[0-9]+\.[0-9]+", string=l))
#     start = millisec(start) - 500
#     end = millisec(end) + 500
#     speaker = re.findall("SPEAKER_[0-9]+", string=l)
#     dzlist.append([start, end, speaker[0]])

#     # cut audio into speaker sections, save each section, transcribe, then delete
#     section = audio[start:end]
#     section.export(str(sectionpath), format="wav")
#     line = model.transcribe(str(sectionpath), language="en")
#     lines.append([start, end, speaker[0], line["text"]])
#     os.remove(str(sectionpath))

# for i in lines:
#     if i[3] != "":
#         print(f"{i[2]}: {i[3]}")


