# Audio-Transcriber
 Program to transcribe audio recordings into text. Output is in the form of a dialog, separated by different speakers.
 
 # Approach
Uses Flask to create a webpage to upload audio files. Audio files are then split according to different speakers using [pyannote.audio](https://github.com/pyannote/pyannote-audio), and then transcribed using [whisper](https://github.com/openai/whisper). Results are then displayed on the webpage
