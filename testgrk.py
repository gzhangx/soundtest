import numpy as np
import matplotlib.pyplot as plt
import wave
import sounddevice as sd
from matplotlib.widgets import SpanSelector

class AudioWindow:
    def __init__(self, data, fs, title_prefix=""):
        self.data = data
        self.fs = fs
        self.title_prefix = title_prefix
        self.fig = None
        self.span = None
        self.create_window()

    def hanning(self, n):
        return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / (n - 1))

    def play_audio(self, data):
        if data.dtype != np.float32:
            data = data.astype(np.float32) / np.max(np.abs(data))
        sd.play(data, self.fs)
        sd.wait()

    def create_window(self):
        # Create time array
        time = np.linspace(0, len(self.data) / self.fs, num=len(self.data))

        # Spectrogram parameters
        n_fft = 1024
        hop_length = 256
        window = self.hanning(n_fft)

        # Calculate spectrogram
        spec = []
        n_frames_spec = (len(self.data) - n_fft) // hop_length + 1
        if n_frames_spec <= 0:
            print("Selection too short for spectrogram analysis. Playing audio only.")
            self.play_audio(self.data)
            return

        for i in range(0, len(self.data) - n_fft, hop_length):
            frame = self.data[i:i + n_fft] * window
            fft_result = np.fft.fft(frame)
            fft_magnitude = np.abs(fft_result[:n_fft//2])
            spec.append(fft_magnitude)
        
        spec = np.array(spec).T
        freqs = np.linspace(0, self.fs/2, n_fft//2)
        times = np.linspace(0, len(self.data)/self.fs, n_frames_spec)

        # Create figure
        self.fig = plt.figure(figsize=(12, 8))
        ax1 = self.fig.add_subplot(2, 1, 1)
        ax2 = self.fig.add_subplot(2, 1, 2)

        # Plot waveform
        ax1.plot(time, self.data)
        ax1.set_title(f'{self.title_prefix}Waveform')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True)

        # Plot spectrogram
        ax2.imshow(20 * np.log10(spec + 1e-10),
                   aspect='auto',
                   origin='lower',
                   extent=[times[0], times[-1], freqs[0], freqs[-1]],
                   cmap='viridis')
        ax2.set_title(f'{self.title_prefix}Spectrogram')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Frequency (Hz)')

        # Selection handler
        def onselect(xmin, xmax):
            start_idx = int(xmin * self.fs)
            end_idx = int(xmax * self.fs)
            if end_idx > len(self.data):
                end_idx = len(self.data)
            if start_idx < 0:
                start_idx = 0
            if end_idx <= start_idx:
                print("Invalid selection range")
                return
            
            print(f"Playing selection from {xmin:.2f}s to {xmax:.2f}s")
            selected_data = self.data[start_idx:end_idx]
            self.play_audio(selected_data)
            # Create new window with the selected section
            new_window = AudioWindow(selected_data, self.fs, f"Selected ({xmin:.2f}-{xmax:.2f}s) ")

        # Add SpanSelector
        self.span = SpanSelector(
            ax1,
            onselect,
            'horizontal',
            useblit=True,
            props=dict(alpha=0.5, facecolor='red'),
            interactive=True
        )

        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.show()

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

# Create the main window
main_window = AudioWindow(audio_data, fs, "Main ")

# Instructions
print("Click and drag on the waveform to select a region to play back and analyze")
print(f"Audio duration: {len(audio_data)/fs:.2f} seconds")
print(f"Sample rate: {fs} Hz")
print(f"Spectrogram parameters: FFT size = 1024, Hop length = 256")

# Start the matplotlib event loop
plt.ion()  # Enable interactive mode
plt.show(block=True)  # Keep the event loop running