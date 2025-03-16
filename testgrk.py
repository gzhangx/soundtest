import numpy as np
import matplotlib.pyplot as plt
import wave
import sounddevice as sd
from matplotlib.widgets import SpanSelector

# Function to play selected audio section and display in new window
def play_and_display(data, fs, start_idx, end_idx):
    selected_data = data[start_idx:end_idx]
    # Normalize for sounddevice
    if selected_data.dtype != np.float32:
        selected_data = selected_data.astype(np.float32) / np.max(np.abs(selected_data))
    sd.play(selected_data, fs)
    
    # Create time array for selection
    selected_time = np.linspace(0, len(selected_data) / fs, num=len(selected_data))
    
    # Calculate spectrogram for selection
    n_fft = 1024
    hop_length = 256
    window = hanning(n_fft)
    spec = []
    n_frames_spec = (len(selected_data) - n_fft) // hop_length + 1
    for i in range(0, len(selected_data) - n_fft, hop_length):
        frame = selected_data[i:i + n_fft] * window
        fft_result = np.fft.fft(frame)
        fft_magnitude = np.abs(fft_result[:n_fft//2])
        spec.append(fft_magnitude)
    
    spec = np.array(spec).T
    freqs = np.linspace(0, fs/2, n_fft//2)
    times = np.linspace(0, len(selected_data)/fs, n_frames_spec)
    
    # Create new figure for selection
    fig_sel, (ax1_sel, ax2_sel) = plt.subplots(2, 1, figsize=(8, 6))
    
    # Plot selected waveform
    ax1_sel.plot(selected_time, selected_data)
    ax1_sel.set_title('Selected Waveform')
    ax1_sel.set_xlabel('Time (s)')
    ax1_sel.set_ylabel('Amplitude')
    ax1_sel.grid(True)
    
    # Plot selected spectrogram
    ax2_sel.imshow(20 * np.log10(spec + 1e-10),
                   aspect='auto',
                   origin='lower',
                   extent=[times[0], times[-1], freqs[0], freqs[-1]],
                   cmap='viridis')
    ax2_sel.set_title('Selected Spectrogram')
    ax2_sel.set_xlabel('Time (s)')
    ax2_sel.set_ylabel('Frequency (Hz)')
    
    plt.tight_layout()
    fig_sel.show()
    
    sd.wait()  # Wait for playback to finish

# Function called when selection is made
def onselect(xmin, xmax):
    start_idx = int(xmin * fs)
    end_idx = int(xmax * fs)
    print(f"Playing selection from {xmin:.2f}s to {xmax:.2f}s")
    play_and_display(audio_data, fs, start_idx, end_idx)

# Load the WAV file using wave
filename = "h123.wav"
with wave.open(filename, 'rb') as wf:
    fs = wf.getframerate()
    n_channels = wf.getnchannels()
    sample_width = wf.getsampwidth()
    n_frames = wf.getnframes()
    
    # Read raw data
    raw_data = wf.readframes(n_frames)
    
    # Convert raw bytes to numpy array based on sample width
    if sample_width == 1:
        audio_data = np.frombuffer(raw_data, dtype=np.uint8) - 128
    elif sample_width == 2:
        audio_data = np.frombuffer(raw_data, dtype=np.int16)
    else:
        raise ValueError("Only 8-bit and 16-bit WAV files are supported")

    # Convert stereo to mono if needed
    if n_channels > 1:
        audio_data = audio_data.reshape(-1, n_channels).mean(axis=1)

# Create time array
time = np.linspace(0, len(audio_data) / fs, num=len(audio_data))

# Spectrogram parameters
n_fft = 1024
hop_length = 256

# Simple Hanning window implementation
def hanning(n):
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / (n - 1))

window = hanning(n_fft)

# Calculate spectrogram using numpy.fft
spec = []
n_frames_spec = (len(audio_data) - n_fft) // hop_length + 1
for i in range(0, len(audio_data) - n_fft, hop_length):
    frame = audio_data[i:i + n_fft] * window
    fft_result = np.fft.fft(frame)
    fft_magnitude = np.abs(fft_result[:n_fft//2])
    spec.append(fft_magnitude)

spec = np.array(spec).T
freqs = np.linspace(0, fs/2, n_fft//2)
times = np.linspace(0, len(audio_data)/fs, n_frames_spec)

# Create main figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Plot waveform
ax1.plot(time, audio_data)
ax1.set_title('Waveform')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Amplitude')
ax1.grid(True)

# Plot spectrogram
ax2.imshow(20 * np.log10(spec + 1e-10),
           aspect='auto',
           origin='lower',
           extent=[times[0], times[-1], freqs[0], freqs[-1]],
           cmap='viridis')
ax2.set_title('Spectrogram')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Frequency (Hz)')

# Add SpanSelector to waveform plot
span = SpanSelector(
    ax1,
    onselect,
    'horizontal',
    useblit=True,
    props=dict(alpha=0.5, facecolor='red'),
    interactive=True
)

plt.tight_layout()

# Instructions
print("Click and drag on the waveform to select a region to play back")
print(f"Audio duration: {len(audio_data)/fs:.2f} seconds")
print(f"Sample rate: {fs} Hz")
print(f"Spectrogram parameters: FFT size = {n_fft}, Hop length = {hop_length}")

plt.show()