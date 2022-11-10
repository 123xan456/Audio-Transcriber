# Audio-Transcriber
 Program to transcribe audio recordings into text. Output is in the form of a dialog, separated by different speakers. Additionally outputs a 100 word summmary of the entire clip
 
 # Approach
Uses Flask to create a webpage to upload audio files. 

1) [NVIDIA's NeMo](https://github.com/NVIDIA/NeMo) identifies different speakers in the audio file, and returns the timeframes in which they speak. 

2) Audio segments are then cut according to each speaker from the original audio file using [pydub](https://github.com/jiaaro/pydub).

3) Segments are then transcribed using [whisper](https://github.com/openai/whisper). Results are then displayed on the webpage in the form of a dialogue

4) [TextRank](https://github.com/davidadamojr/TextRank) is used to output a 100-word summary of the audio clip using the transcribed text. Longer clips are given a 100 word summary every 10 minutes

setup.txt is added to list all dependencies needed
