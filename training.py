# Treinamento do classificador
# usando o dataset 'fma_tiny'

# Acesso dos arquivos
import os
# Acesso dos arquivos .csv
import csv
# Contabilizacao o algoritmo
import time
# Aleatorizacao do dataset
# import random
# Manipulacao de estruturas de dados
import numpy as np
# Alguns utilitarios
import sklearn as sk
# Implementacao do classificador
import tensorflow as tf

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Constantes do programa
GENRES = {'Electronic'    : 0,
          'Experimental'  : 1,
          'Folk'          : 2,
          'Hip-Hop'       : 3,
          'Instrumental'  : 4,
          'International' : 5,
          'Pop'           : 6,
          'Rock'          : 7}

LABEL_FILE  = 'data/track_labels_small.csv'
DATASET_DIR = 'data/fma_small_mel/'

TRACKS_PER_GENRE = 20
DATASET_SIZE = TRACKS_PER_GENRE * len(GENRES)
TRAIN_SIZE = DATASET_SIZE // 2
VALID_SIZE = DATASET_SIZE // 4
TEST_SIZE  = DATASET_SIZE // 4
NORMALIZE_METHOD = 0

BATCH_SIZE = 0.05 * TRAIN_SIZE
TRACK_SHAPE = (128, 1288)
TRACK_RESHAPE = (64, 320)
EPOCHS = 10

# Obtem os labels de todas as musicas do fma_small
def get_track_labels():
  file = csv.reader(open(LABEL_FILE, 'r'))
  all_labels = {int(rows[0]) : GENRES[rows[1]] for rows in file}
  return all_labels

# Diminui as dimensoes do espectrograma
# Para diminuir carga para a memoria
def reshape_track(track):
  oldx, oldy = TRACK_SHAPE[0], TRACK_SHAPE[1]
  newx, newy = TRACK_RESHAPE[0], TRACK_RESHAPE[1]
  
  aux = np.zeros((newx, oldy))
  newls = np.linspace(0, 1, newx)
  oldls = np.linspace(0, 1, oldx)
  for col in range(oldy):
    aux[:,col] = np.interp(newls, oldls, track[:,col])
  
  newtrack = np.zeros((newx, newy))
  newls = np.linspace(0, 1, newy)
  oldls = np.linspace(0, 1, oldy)
  for lin in range(newx):
    newtrack[lin,:] = np.interp(newls, oldls, aux[lin,:])
  
  return newtrack

# Dataset gerado aleatoriamente, com
# as musicas contidas no fma_small,
# ignorando as musicas 'bugadas'
def get_dataset_fma_tiny(all_labels):
  IGNORE = [  1486,   5574,  65753,  80391,  98558,  98559,
             98560,  98565,  98566,  98567,  98568,  98569,
             98571,  99134, 105247, 107535, 108924, 108925,
            126981, 127336, 133297, 143992]
  num_genres = [0 for i in GENRES]
  
  # Pega o nome de todas as musicas
  filenames = []
  for (_, _, file) in os.walk(DATASET_DIR):
    filenames.extend(file)
  
  # Embaralha a lista dos nomes
  filenames = sk.utils.shuffle(filenames)
  
  # Percorre a lista ate obter o novo dataset completo
  train_set = []
  train_label = []
  valid_set = []
  valid_label = []
  test_set = []
  test_label = []
  counter = 0
  for file in filenames:
    # Pula as musicas da 'lista negra'
    if int(file[:-4]) in IGNORE:
      continue
    
    # Verifca se ja tem todas as musicas de tal genero
    genre = all_labels[int(file[:-4])]
    if num_genres[genre] >= TRACKS_PER_GENRE:
      continue
    
    # Abre arquivo da musica
    track = np.load('%s%s' % (DATASET_DIR, file))
    # Se a musica for faltosa nao reconhecida, anota o nome
    if np.std(track) == 0.0 or track.shape != TRACK_SHAPE:
      print('Found new faulty track: %s' % file)
      continue
    
    # Senao, insere musica no dataset
    # Primeiro enche o conjunto de teste
    if num_genres[genre] < TRACKS_PER_GENRE // 4:
      test_set.append(reshape_track(track))
      test_label.append(all_labels[int(file[:-4])])
    # Depois o conjunto de validacao
    elif num_genres[genre] < TRACKS_PER_GENRE // 2:
      valid_set.append(reshape_track(track))
      valid_label.append(all_labels[int(file[:-4])])
    # E por ultimo o conjunto de treino
    else:
      train_set.append(reshape_track(track))
      train_label.append(all_labels[int(file[:-4])])
    
    num_genres[genre] += 1
    counter += 1
    
    # Dataset esta completo, termina a funcao
    if counter >= DATASET_SIZE:
      break
  
  # return dataset, labels
  return train_set, train_label, valid_set, valid_label, test_set, test_label

# Normaliza os espectrogramas do dataset
def normalize(dataset):
  # Deixa o espectrograma com média em 0 e desvio padrão 1
  if NORMALIZE_METHOD == 1:
    dataset = [(x - np.mean(x)) / np.std(x) for x in dataset]
  
  # Deixa valores do espectrograma entre 0 e 1
  elif NORMALIZE_METHOD == 2:
    min = np.amin(dataset)
    max = np.amax(dataset)
    dataset = [(x - min) / (max - min) for x in dataset]
  
  dataset = np.array(dataset).reshape((-1,) + TRACK_RESHAPE + (1,))
  
  return dataset

# Separa dataset em conjunto de treino e de teste
def split_dataset(dataset, labels):
  train_set = (dataset[0 :   int(len(dataset) * train_ratio)][:][:],
                labels[0 :   int(len( labels) * train_ratio)]      )
  test_set  = (dataset[int(len(dataset) * train_ratio) :    ][:][:],
                labels[int(len( labels) * train_ratio) :    ]      )
  return train_set, valid_set, test_set

# Funcao main
if __name__ == '__main__':
  time_before = time.time()
  
  # Dicionario de numero da musica : classe
  all_labels = get_track_labels()
  # Conjuntos das musicas e de suas classes
  train_set, train_label, valid_set, valid_label, test_set, test_label = get_dataset_fma_tiny(all_labels)
  # Normaliza o dataset
  train_set = normalize(train_set)
  valid_set = normalize(valid_set)
  test_set = normalize(test_set)
  
  dataset_init_time = time.time() - time_before
  
  # Criacao do modelo de rede neural
  model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(TRACK_RESHAPE + (1,))),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(len(GENRES), activation='softmax')
  ])
  
  model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  
  model.summary()
  
  history = model.fit(train_set, train_label,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      steps_per_epoch=int(np.ceil(TRAIN_SIZE / BATCH_SIZE)),
                      validation_data=(valid_set, valid_label),
                      validation_steps=int(np.ceil(VALID_SIZE / BATCH_SIZE)))
  
  print('\n# History dict:', history.history)
  
  import matplotlib.pyplot as plt
  
  acc = history.history['acc']
  val_acc = history.history['val_acc']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs_range = range(EPOCHS)

  plt.figure(figsize=(8, 8))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  # plt.savefig('./foo.png')
  plt.show()
  
  print('\n# Evaluate on test data')
  results = model.evaluate(test_set, test_label, batch_size=BATCH_SIZE)
  print('Test loss, Test acc:', results)
