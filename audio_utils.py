# audio_utils.py
import pyaudio
import numpy as np
import wave

RATE = 8000
CHANNELS = 1
FORMAT = pyaudio.paInt16
BYTES_PER_SAMPLE = 2 #16 bit
def record_to_numpy(record_seconds=5, chunk=1024):
    """
    Record audio into a NumPy array with the lowest rate (8000 Hz) and format (paInt8).
    """
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       input=True,
                       frames_per_buffer=chunk)

    print("Recording started...")
    total_samples = int(RATE * record_seconds)
    buffers = []  # List to store raw audio buffers

    # Record chunks into buffers
    for _ in range(0, int(total_samples / chunk)):
        data = stream.read(chunk)
        buffers.append(np.frombuffer(data, dtype=np.int8))

    # Handle remainder if total_samples isn’t a multiple of chunk
    remainder = total_samples % chunk
    if remainder:
        data = stream.read(remainder)
        buffers.append(np.frombuffer(data, dtype=np.int8))

    # Concatenate all buffers into a single NumPy array
    audio_data = np.concatenate(buffers)

    print("Recording finished.")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    return audio_data

def save_numpy_to_wav(audio_data, filename):
    """
    Save a NumPy array as a WAV file.
    """
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(BYTES_PER_SAMPLE)  # 1 byte for int8
    wf.setframerate(RATE)
    audio_buffer = audio_data.tobytes()
    wf.writeframes(audio_buffer)
    wf.close()
    print(f"Audio saved to {filename}")

def load_wav_to_numpy(filename):
    """
    Load a WAV file into a NumPy array using wave and numpy.
    """
    with wave.open(filename, 'rb') as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()

        raw_data = wf.readframes(n_frames)

        if sample_width == 1:
            dtype = np.int8  # 8-bit signed
        elif sample_width == 2:
            dtype = np.int16
        elif sample_width == 4:
            dtype = np.int32
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        audio_data = np.frombuffer(raw_data, dtype=dtype)

        if channels > 1:
            audio_data = audio_data.reshape(-1, channels)

        print(f"Loaded {filename}")
        print(f"Sample Rate: {sample_rate} Hz")
        print(f"Channels: {channels}")
        print(f"Audio Shape: {audio_data.shape}")

        return sample_rate, audio_data