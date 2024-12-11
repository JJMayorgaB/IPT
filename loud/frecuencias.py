import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from scipy.fft import fft
from tkinter import filedialog
from tkinter import Tk

# Usar Tkinter para abrir un cuadro de diálogo de selección de archivo
root = Tk()
root.withdraw()  # Oculta la ventana principal de Tkinter

# Seleccionar el archivo .mp3 o .aac desde el explorador de archivos
file_path = filedialog.askopenfilename(title="Selecciona un archivo de audio", filetypes=[("Audio Files", "*.mp3;*.aac;*.wav")])

# Cargar el archivo de audio
audio = AudioSegment.from_file(file_path, format="mp3" if file_path.endswith(".mp3") else "aac")

# Convertir el audio a mono (un solo canal) si es estéreo
audio = audio.set_channels(1)

# Convertir el audio a una frecuencia de muestreo estándar (por ejemplo, 44100 Hz)
audio = audio.set_frame_rate(44100)

# Obtener las muestras de audio en formato de array
samples = np.array(audio.get_array_of_samples())

# Normalizar los valores de las muestras (si es necesario)
samples = samples / np.max(np.abs(samples))  # Normalización a [-1, 1]

# Realizar la Transformada Rápida de Fourier (FFT)
n = len(samples)  # Número de muestras
frequencies = np.fft.fftfreq(n, d=1/44100)  # Generar las frecuencias correspondientes
fft_result = fft(samples)  # Realizar la FFT

# Tomar la magnitud de la FFT (solo la parte positiva, porque es simétrica)
fft_magnitude = np.abs(fft_result)[:n//2]
frequencies = frequencies[:n//2]  # Solo las frecuencias positivas

# Graficar el espectro de frecuencias
plt.figure(figsize=(10, 6))
plt.plot(frequencies, fft_magnitude)
plt.title("Espectro de Frecuencias")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.grid(True)

# Establecer los límites del eje X entre 300 y 1500 Hz
plt.xlim(0, 1500)

# Mostrar la gráfica
plt.show()
