import sounddevice as sd
import numpy as np
import soundfile as sf
import asyncio
from faster_whisper import WhisperModel

VAD_THRESHOLD = 0.01  

async def capture_audio(duration=5, filename="audio.flac"):
    fs = 16000 
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.float32)
    sd.wait()  
    print("Recording complete.")

    if vad_filter(audio, threshold=VAD_THRESHOLD):
        sf.write(filename, audio, fs, format='FLAC')
        return filename
    else:
        print("No significant audio detected.")
        return None
def vad_filter(audio, threshold=VAD_THRESHOLD):
    return np.max(np.abs(audio)) > threshold

async def transcribe_audio(filename):
    model = WhisperModel("medium", device="cpu", compute_type="int8") 

    segments, info = model.transcribe(filename)
    transcribed_text = ""
    for segment in segments:
        transcribed_text += segment.text + " "

    print("Transcribed Text:", transcribed_text)
    return transcribed_text

async def main():
    audio_file = await capture_audio()

    if audio_file:
        await transcribe_audio(audio_file)

asyncio.run(main())