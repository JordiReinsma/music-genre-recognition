# Preprocessamento dos dados
# mp3 -> espectrograma (imagem em escalas de cinza)

import os
import numpy as np
import librosa

# Faulty MP3 files (https://github.com/mdeff/fma/issues/8).
# MP3 file IDs with 0 second of audio.
IGNORE = [  1486,   5574,  65753,  80391,  98558,  98559,  98560,
           98565,  98566,  98567,  98568,  98569,  98571,  99134,
          105247, 108924, 108925, 126981, 127336, 133297, 143992]

DATASET_DIR = 'data/fma_small/'
OUTPUT_DIR = 'data/fma_small_mel/'

# Pegando o nome de todas as subpastas dentro da pasta do dataset
dirnames = []
for (_, d, _) in os.walk(DATASET_DIR):
  dirnames.extend(d)

for dir in dirnames[:3]:
  # Pegando o nome de todos os arquivos (as musicas de fato)
  # dentro das subpastas
  filenames = []
  for (_, _, f) in os.walk('%s%s/' % (DATASET_DIR, dir)):
    filenames.extend(f)
  
  for file in filenames:
    # Ignora se a musica estiver na lista das defeituosas
    if int(file[:-4]) in IGNORE:
      continue
    
    # Gera espectrograma de cada musica
    y, sr = librosa.load('%s%s/%s' % (DATASET_DIR, dir, file), duration=29.9, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel = librosa.power_to_db(mel, ref=np.max)
    
    # Salva os arquivos de espectrograma em uma unica pasta
    x = np.array(mel)
    np.save('%s%s.npy' % (OUTPUT_DIR, file[:-4]), x)
  
  print(dir + ' done!')
