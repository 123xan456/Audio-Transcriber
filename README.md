# Audio-Transcriber
 Program to transcribe audio recordings into text. Output is in the form of a dialog, separated by different speakers.
 
 # Approach
Uses Flask to create a webpage to upload audio files. [pyannote.audio](https://github.com/pyannote/pyannote-audio) allows for the different speakers to be identified in the audio file, and returns the timeframes in which they speak. Audio segments are then cut according to each speaker from the original audio file, and then transcribed using [whisper](https://github.com/openai/whisper). Results are then displayed on the webpage in the form of a dialogue
