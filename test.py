import pyaudio
import wave

# Audio recording parameters
FORMAT = pyaudio.paInt16    # 8-bit format (lowest common format)
CHANNELS = 1               # Mono
RATE = 8000                # 8000 Hz (lowest common sample rate)
CHUNK = 1024               # Buffer size
RECORD_SECONDS = 5         # Recording duration
OUTPUT_FILE = "output.wav" # Output WAV file

def record_to_wav():
    """
    Record audio from the microphone and save it to a WAV file.
    """
    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Open stream
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    print("Recording started...")

    # Collect audio data
    frames = []
    for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording finished.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save to WAV file
    with wave.open(OUTPUT_FILE, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 1 byte for paInt8
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    record_to_wav()