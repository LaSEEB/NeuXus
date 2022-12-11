import lib.python.detect as detect
from lib.python.find_files import find_files
import tensorflow as tf  # Install tensorflow library
import pandas as pd
import pickle


dset = ''
lfol = 'data/detected/'
sfol = 'model/'
pairs_train = {'set': [dset], 'fol': [lfol], 'sub': ['patient005'], 'ses': ['interictal'], 'task': ['eegcalibration'], 'chan': ['ecg'], 'cor':['fiNeXoff'], 'det':['qrs']}
pairs_valid = {'set': [dset], 'fol': [lfol], 'sub': ['patient009'], 'ses': ['interictal'], 'task': ['eegcalibration'], 'chan': ['ecg'], 'cor':['fiNeXoff'], 'det':['qrs']}

# Find files
files_train = find_files(pairs_train)
files_valid = find_files(pairs_valid)

io = detect.Rio()
train_ecgs, train_gndts = io.load(lfol, [file['filename'] for file in files_train])
valid_ecgs, valid_gndts = io.load(lfol, [file['filename'] for file in files_valid])

trainer = detect.Rtrain()
n_batch = 256
n_timesteps = 500  # Originally: 1000  (at 250 Hz => 4 seconds)
n_input_dim = 1
val_steps = 2
activ_fun = 'sigmoid'
loss = 'binary_crossentropy'
optim = 'adam'
metrics = ['acc', tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision()]
epochs = 10  # Originally: 40  (Lowered to 10 by seeing in the train accuracy reaching a plateau soon after 10)
steps_per_epoch = 40
train_gen = trainer.pick(train_ecgs, train_gndts, n_timesteps, n_batch)
valid_gen = trainer.pick(valid_ecgs, valid_gndts, n_timesteps, n_batch)
model = trainer.create(n_timesteps, n_input_dim, activ_fun, loss, optim, metrics)
model, history, fh = trainer.train(model, train_gen, valid_gen, val_steps, epochs, steps_per_epoch)
weights = trainer.extract(model, units=64, timesteps=n_timesteps)
pickle.dump(weights, open(sfol + 'weights.pkl', 'wb'))  # Save as .pkl (NeuXus loads this version)
# io.save(sfol, 'model.h5', model, history, fh, weights)  # Save as .h5 (optional)
