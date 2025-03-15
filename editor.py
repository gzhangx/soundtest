import numpy as np
import sounddevice as sd
import wave
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor

class FFTEditor:
    def __init__(self, audio_data, sample_rate):
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.n = len(audio_data)
        self.fft_data = np.fft.fft(audio_data)
        self.freq = np.fft.fftfreq(self.n, 1/sample_rate)
        self.magnitude = np.abs(self.fft_data)
        self.modified_fft = self.fft_data.copy()
        self.fig = None
        self.ax = None



    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        # Get clicked frequency and magnitude
        clicked_freq = event.xdata
        clicked_mag = event.ydata

        # Find closest frequency index
        idx = np.abs(self.freq - clicked_freq).argmin()
        if idx > self.n//2:  # Only modify positive frequencies
            return

        # Update magnitude at this point
        new_magnitude = clicked_mag
        current_magnitude = np.abs(self.modified_fft[idx])

        # Calculate scaling factor
        if current_magnitude != 0:
            scale = new_magnitude / current_magnitude
        else:
            scale = new_magnitude

        # Apply to both positive and negative frequencies
        self.modified_fft[idx] *= scale
        self.modified_fft[self.n - idx] = np.conj(self.modified_fft[idx])

        # Update plot
        self.ax.clear()
        self.ax.plot(self.freq[:self.n//2], np.abs(self.modified_fft)[:self.n//2])
        self.ax.set_title('Click to modify FFT points (Right-click to finish)')
        self.ax.set_xlabel('Frequency (Hz)')
        self.ax.set_ylabel('Magnitude')
        self.ax.grid()
        self.fig.canvas.draw()

    def on_right_click(self, event):
        if event.button == 3:  # Right click
            plt.close(self.fig)

    def edit_fft(self):
        # Create interactive plot
        self.fig, self.ax = plt.subplots(figsize=(12, 6))

        # Plot initial FFT
        self.ax.plot(self.freq[:self.n//2], self.magnitude[:self.n//2])
        self.ax.set_title('Click to modify FFT points (Right-click to finish)')
        self.ax.set_xlabel('Frequency (Hz)')
        self.ax.set_ylabel('Magnitude')
        self.ax.grid()

        # Add cursor
        cursor = Cursor(self.ax, useblit=True, color='red', linewidth=1)

        # Connect event handlers
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('button_press_event', self.on_right_click)

        plt.show()

        return self.modified_fft

    def play_modified_audio(self):
        modified_audio = np.fft.ifft(self.modified_fft)
        modified_audio = np.real(modified_audio)
        modified_audio = np.clip(modified_audio, -32768, 32767).astype(np.int16)

        print("Playing modified audio...")
        sd.play(modified_audio, self.sample_rate)
        sd.wait()

def read_wav_file(filename):
        with wave.open(filename, 'rb') as wav_file:
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            framerate = wav_file.getframerate()
            n_frames = wav_file.getnframes()

            raw_data = wav_file.readframes(n_frames)

            if sample_width == 1:
                data = np.frombuffer(raw_data, dtype=np.uint8)
                data = data - 128
            elif sample_width == 2:
                data = np.frombuffer(raw_data, dtype=np.int16)
            else:
                raise ValueError("Only supports 8 or 16 bit audio")

            if n_channels == 2:
                data = data[::2]

            return data, framerate

def main():
    #filename = input("Enter the path to your WAV file: ")
    filename = 'h123.wav'

    try:
        # Create editor instance
        print("editor created")
        audio_data, sample_rate = read_wav_file(filename)
        editor = FFTEditor(audio_data, sample_rate)

        print(f"Sample rate: {sample_rate} Hz")
        print(f"Duration: {len(audio_data)/sample_rate:.2f} seconds")

        # Edit FFT interactively
        print("Left-click to adjust FFT points, right-click when finished")
        modified_fft = editor.edit_fft()

        # Play result
        editor.play_modified_audio()

    except FileNotFoundError:
        print("File not found!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()