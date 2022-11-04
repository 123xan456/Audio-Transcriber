# Audio-Transcriber
 Program to transcribe audio recordings into text. Output is in the form of a dialog, separated by different speakers. Additionally outputs a 100 word summmary of the entire clip
 
 # Approach
Uses Flask to create a webpage to upload audio files. 

1) [pyannote.audio](https://github.com/pyannote/pyannote-audio) allows for the different speakers to be identified in the audio file, and returns the timeframes in which they speak. 

2) Audio segments are then cut according to each speaker from the original audio file using [pydub](https://github.com/jiaaro/pydub).

3) Segments are then transcribed using [whisper](https://github.com/openai/whisper). Results are then displayed on the webpage in the form of a dialogue

4) [TextRank](https://github.com/davidadamojr/TextRank) is used to output a 100-word summary of the entire audio clip using the transcribed text.
