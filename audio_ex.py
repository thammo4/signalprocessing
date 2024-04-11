import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;

import seaborn as sns;

from glob import glob;

import librosa;
import librosa.display;


# import IPython.display as ipd;
import subprocess;

from itertools import cycle;

sns.set_theme(style='white', palette=None);
color_pal = plt.rcParams['axes.prop_cycle'].by_key()['color'];
color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color']);


#
# Index the audio files
#

audio_files = glob('./RAVDESS/Actor_*/*.wav');



#
# Play an audio file as an example
#

audio_file_ex = audio_files[10];
subprocess.run(['afplay', audio_file_ex]);


#
# Load audio file data with librosa
#

y, sr = librosa.load(audio_file_ex);

print(f'y: {y[:10]}');
print(f'shape y: {y.shape}');
print(f'sample rate: {sr}');


#
# Plot the audio example
#

plt.figure(figsize=(10,5));
pd.Series(y).plot(lw=1, title='Raw Audio Example', color=color_pal[0]);

# plt.show();
plt.savefig('raw_audio_example.png', bbox_inches='tight'); plt.close();


#
# Plot trimmed version of the audio example
#

y_trimmed, _ = librosa.effects.trim(y);

plt.figure(figsize=(10,5));
pd.Series(y_trimmed).plot(lw=1, title='Trimmed Audio Example', color=color_pal[1]);

# plt.show();
plt.savefig('naive_trimmed_audio_example.png', bbox_inches='tight'); plt.close();


#
# Plot trimmed audio exmaple with `top_db` changed to 20
#

y_trimmed, _ = librosa.effects.trim(y, top_db=20);

plt.figure(figsize=(10,5));
pd.Series(y_trimmed).plot(lw=1, title='Raw Audio Trimmed Example', color=color_pal[1]);

# plt.show();
plt.savefig('trimmed_audio_example.png', bbox_inches='tight'); plt.close();


#
# Plot subset of the raw audio file
#

plt.figure(figsize=(10,5));
pd.Series(y[30000:30500]).plot(lw=1, title='Audio Subset 30k-31k', color=color_pal[2]);

# plt.show();
plt.savefig('subset_audio_example.png', bbox_inches='tight'); plt.close();



#
# Apply Fourier Transform to identify frequency components
#

D = librosa.stft(y);
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max);


#
# Construct Spectogram
#

fig, ax = plt.subplots(figsize=(10,5));
img = librosa.display.specshow(S_db, x_axis='time', y_axis='log', ax=ax);
ax.set_title('Spectogram Audio File Example', fontsize=20);
fig.colorbar(img, ax=ax, format=f'%0.2f');

# plt.show();
plt.savefig('spectogram_audio_example.png', bbox_inches='tight'); plt.close();



#
# Construct Mel Spectogram (Mel=Melodic)
#

S_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128); print(f'S_mel shape: {S_mel.shape}');
S_mel_db = librosa.amplitude_to_db(S_mel, ref=np.max);
fig, ax = plt.subplots(figsize=(10,5));
img_mel = librosa.display.specshow(S_mel_db, x_axis='time', y_axis='log', ax=ax);
ax.set_title('Mel Spectogram of Audio Example', fontsize=20);
fig.colorbar(img_mel, ax=ax, format=f'%0.2f');

# plt.show();
plt.savefig('mel_spectogram_audio_example.png', bbox_inches='tight'); plt.close();



#
# Construct Mel Spectogram Using Twice As Many Melodies
#

S_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256); print(f'S_mel shape: {S_mel.shape}');
S_mel_db = librosa.amplitude_to_db(S_mel, ref=np.max);
fig, ax = plt.subplots(figsize=(10,5));
img_mel = librosa.display.specshow(S_mel_db, x_axis='time', y_axis='log', ax=ax);
ax.set_title('Mel Spectogram of Audio Example', fontsize=20);
fig.colorbar(img_mel, ax=ax, format=f'%0.2f');

# plt.show();
plt.savefig('mel_spectogram2_audio_example.png', bbox_inches='tight'); plt.close();