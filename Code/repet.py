
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import librosa.display
import scipy
import pandas


# Load vocals
y, sr = librosa.load('name.wav', offset=0,duration=10)

# Calculate the spectrogram magnitude and phase
S_full, phase = librosa.magphase(librosa.stft(y))



idx = slice(*librosa.time_to_frames([0, 5], sr=sr))
plt.figure(figsize=(12, 4))
librosa.display.specshow(librosa.amplitude_to_db(S_full[:, idx], ref=np.max),
                         y_axis='log', x_axis='time', sr=sr)
plt.colorbar()
plt.tight_layout()


S_filter = librosa.decompose.nn_filter(S_full,
                                       aggregate=np.median,
                                       metric='cosine',
                                       width=int(librosa.time_to_frames(2, sr=sr)))




S_filter = np.minimum(S_full, S_filter)


margin_i, margin_v = 2, 10
power = 2

mask_i = librosa.util.softmask(S_filter,
                               margin_i * (S_full - S_filter),
                               power=power)

mask_v = librosa.util.softmask(S_full - S_filter,
                               margin_v * S_filter,
                               power=power)



S_foreground = mask_v * S_full
S_background = mask_i * S_full
S_foreground1 = S_foreground * phase
S_background1 = S_background * phase

voc=librosa.core.istft(S_foreground1 )
mus=librosa.core.istft(S_background1 )
librosa.output.write_wav('music.wav', mus, sr=sr)
librosa.output.write_wav('vocal.wav', voc, sr=sr)

# temp=S_foreground
# for i in range(500,1025):
#   S_foreground[i,:]=0

# for j in range(0,1):
#   S_foreground[j,:]=0

# tempo=temp-S_foreground

# voc=librosa.core.istft(S_foreground*phase)
# mus=librosa.core.istft(S_background1)
# voc1=librosa.core.istft((tempo)*phase)

# librosa.output.write_wav('gam1.wav', mus, sr=sr)
# librosa.output.write_wav('gav1.wav', voc, sr=sr)
# librosa.output.write_wav('gav2.wav', voc1, sr=sr)

plt.figure(figsize=(20, 8))
plt.subplot(3, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(S_full[:, idx], ref=np.max),
                         y_axis='log', sr=sr)
plt.title('Full spectrum')
plt.colorbar()

plt.subplot(3, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(S_background[:, idx], ref=np.max),
                         y_axis='log', sr=sr)
plt.title('Background')
plt.colorbar()

plt.subplot(3, 1, 3)
librosa.display.specshow(librosa.amplitude_to_db(S_foreground[:, idx], ref=np.max),
                         y_axis='log', x_axis='time', sr=sr)
plt.title('Foreground')
plt.colorbar()
plt.tight_layout()
plt.show()

