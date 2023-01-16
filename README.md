# Audio-Transcriber
Program to transcribe audio recordings into text. Output is in the form of a dialog, separated by different speakers. It can also provide summaries for the transcriptions.

# Requirements
Program has been tested only on the following environment:
- Oracle Linux 8
- Python 3.9.7

# Installation
## Installing ffmpeg 
```
sudo yum install ffmpeg ffmpeg-devel
```

## Installing pyaudio
```
sudo yum install portaudio.x86_64 portaudio-devel.x86_64

```

## Installing pip packages
```
pip install -r requirements.txt
```

# Approach
Uses Flask to create a webpage to upload audio files. 

1. [pyannote.audio](https://github.com/pyannote/pyannote-audio) allows for the different speakers to be identified in the audio file, and returns the timeframes in which they speak. 

2. Audio segments are then cut according to each speaker from the original audio file

3. Segments are then transcribed using [whisper](https://github.com/openai/whisper). Results are then displayed on the webpage in the form of a dialogue

# TODO
1. Get rid of logging done by NVIDIA Nemo
2. Update the webpage to display the transcription according to the speakers
3. Update the webpage to display the summarization
4. Add loading screen or bar when the application is processing
5. Use tempfiles instead of saving and reading from an actual file
6. Add more logging to the program
7. Add unittests to the program