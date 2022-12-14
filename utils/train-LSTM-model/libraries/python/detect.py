import numpy as np
from wfdb.processing import normalize_bound
import lib.python.eeglab as eeglab
from os import listdir, path
from matplotlib import pyplot as plt
import pickle
# from numba import jit
# from numba.experimental import jitclass

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.engine import data_adapter


class Rio:
    def __init__(self):
        pass

    @staticmethod
    def load(folder, ecg_chan=31):
        ecgs = []
        gndts = []
        files = [f for f in listdir(folder) if path.isfile(path.join(folder, f))]
        for file in files:
            EEG = eeglab.tools.load(folder + file)
            # ecg = EEG.data[ecg_chan, :].astype(np.float64)
            ecg = EEG.data[0, :].astype(np.float64)
            events = EEG.event[0]
            gndt = np.zeros(len(ecg), dtype=int)
            for e in range(len(events)):
                if events[e].type == 'QRSi':
                    # gndt[events[e].latency[0, 0] - 1] = 1
                    gndt[events[e].latency[0, 0]] = 1
            ecgs.append(ecg)
            gndts.append(gndt)
        return ecgs, gndts

    @staticmethod
    def save(folder, file, model, history, fh, weights):
        model.save(folder + file)
        pickle.dump(history.history, open(folder + 'history', 'wb'))
        pickle.dump(weights, open(folder + 'weights.pkl', 'wb'))
        fh.savefig(folder + 'train_val_metrics.png')
        # with open(folder + '/history', 'wb') as file_pi:
            # pickle.dump(history.history, file_pi)


class Rtrain:
    def __init__(self):
        pass

    def pick(self, signals, gndts, win_size, batch_size, seed=0):
        rng = np.random.default_rng(seed)
        while True:
            X = []
            y = []
            while len(X) < batch_size:
                # idx = rng.random.randint(0, len(signals))
                idx = rng.integers(0, len(signals))
                sig = signals[idx]
                gndt = gndts[idx]
                # Select one window
                # beg = rng.random.randint(sig.shape[0] - win_size + 1)
                beg = rng.integers(sig.shape[0] - win_size + 1)
                end = beg + win_size
                gndt_win = gndt[beg: end].copy()
                if any(gndt_win):
                    gndt_win_ids = np.asarray(np.where(gndt_win))[0]
                    extra_gndt_win_ids = np.concatenate(
                        [gndt_win_ids + 1, gndt_win_ids + 2, gndt_win_ids - 1, gndt_win_ids - 2])
                    extra_gndt_win_ids = extra_gndt_win_ids[
                        (extra_gndt_win_ids >= 0) & (extra_gndt_win_ids < len(gndt_win))]
                    np.put(gndt_win, extra_gndt_win_ids, 1)
                    # Select data for window and normalize it (-1, 1)
                    sig_win = normalize_bound(sig[beg:end], lb=-1, ub=1)
                    X.append(sig_win)
                    y.append(gndt_win)
            X = np.asarray(X)
            y = np.asarray(y)
            X = X.reshape(X.shape[0], X.shape[1], 1)
            y = y.reshape(y.shape[0], y.shape[1], 1).astype(int)
            yield (X, y)

    def create(self, n_timesteps, n_input_dim, activ_fun, loss, optim, metrics):
        # layers = self.layers
        # tf = self.tf
        # Create model and add layers to it
        model = tf.keras.Sequential()
        model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True), input_shape=(n_timesteps, n_input_dim)))
        model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
        model.add(layers.Dense(1, activation=activ_fun))
        # Add custom train step
        model.train_step = self.peek_call(model)
        ## Compile model
        # model.compile(loss=loss, optimizer=optim, metrics=metrics)
        # Debugging:
        model.compile(loss=loss, optimizer=optim, metrics=metrics, run_eagerly=True)
        return model

    def peek_call(self, model):
        original_train_step = model.train_step

        def peek(original_data):
            # data_adapter = self.data_adapter
            # Basically copied one-to-one from https://git.io/JvDTv
            data = data_adapter.expand_1d(original_data)
            x, y_true, w = data_adapter.unpack_x_y_sample_weight(data)
            y_pred = model(x, training=True)
            result = original_train_step(original_data)
            # Add anything here for on_train_batch_end-like behavior
            result['pred'] = y_pred
            result['inputs'] = x
            result['y_true'] = y_true
            return result

        return peek

    def train(self, model: tf.keras.Sequential, train_gen, val_gen, val_steps, epochs, train_steps):
        # Train model
        history = model.fit(train_gen,
                            steps_per_epoch=train_steps,
                            epochs=epochs,
                            use_multiprocessing=False,
                            validation_data=val_gen,
                            validation_steps=val_steps,
                            callbacks=[R_plot_training()])
        model.summary()
        print(history.history['loss'])
        print(history.history['acc'])
        fh1 = plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'], '--')
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'], '--')
        plt.legend(['loss', 'val_loss', 'accuracy', 'val_acc'], loc='upper right')
        plt.show(block=False)
        return model, history, fh1

    @staticmethod
    def extract_lstm_weights(model, layer, param, lims):
        f = model.layers[layer].weights[param][:, lims[0]:lims[1]].numpy()
        i = model.layers[layer].weights[param][:, lims[1]:lims[2]].numpy()
        l = model.layers[layer].weights[param][:, lims[2]:lims[3]].numpy()
        o = model.layers[layer].weights[param][:, lims[3]:lims[4]].numpy()
        return f, i, l, o

    @staticmethod
    def extract_lstm_biases(model, layer, param, lims):
        f = model.layers[layer].weights[param][lims[0]:lims[1]].numpy()
        i = model.layers[layer].weights[param][lims[1]:lims[2]].numpy()
        l = model.layers[layer].weights[param][lims[2]:lims[3]].numpy()
        o = model.layers[layer].weights[param][lims[3]:lims[4]].numpy()
        return f, i, l, o

    @staticmethod
    def extract_dense_weights(model, layer, param):
        return np.transpose(model.layers[layer].weights[param].numpy())

    @staticmethod
    def extract_dense_biases(model, layer, param):
        return model.layers[layer].weights[param].numpy()

    def extract(self, model, units=64, timesteps=1000):
        lims = [lim * units for lim in range(5)]
        l = 0
        wxi_fw1, wxf_fw1, wxl_fw1, wxo_fw1 = Rtrain.extract_lstm_weights(model, l, 0, lims)
        whi_fw1, whf_fw1, whl_fw1, who_fw1 = Rtrain.extract_lstm_weights(model, l, 1, lims)
        bi_fw1, bf_fw1, bl_fw1, bo_fw1 = Rtrain.extract_lstm_biases(model, l, 2, lims)
        wxi_bw1, wxf_bw1, wxl_bw1, wxo_bw1 = Rtrain.extract_lstm_weights(model, l, 3, lims)
        whi_bw1, whf_bw1, whl_bw1, who_bw1 = Rtrain.extract_lstm_weights(model, l, 4, lims)
        bi_bw1, bf_bw1, bl_bw1, bo_bw1 = Rtrain.extract_lstm_biases(model, l, 5, lims)

        l = 1
        wxi_fw2, wxf_fw2, wxl_fw2, wxo_fw2 = Rtrain.extract_lstm_weights(model, l, 0, lims)
        whi_fw2, whf_fw2, whl_fw2, who_fw2 = Rtrain.extract_lstm_weights(model, l, 1, lims)
        bi_fw2, bf_fw2, bl_fw2, bo_fw2 = Rtrain.extract_lstm_biases(model, l, 2, lims)
        wxi_bw2, wxf_bw2, wxl_bw2, wxo_bw2 = Rtrain.extract_lstm_weights(model, l, 3, lims)
        whi_bw2, whf_bw2, whl_bw2, who_bw2 = Rtrain.extract_lstm_weights(model, l, 4, lims)
        bi_bw2, bf_bw2, bl_bw2, bo_bw2 = Rtrain.extract_lstm_biases(model, l, 5, lims)

        dense_w = Rtrain.extract_dense_weights(model, 2, 0)
        dense_b = Rtrain.extract_dense_biases(model, 2, 1)

        weight_vars = [wxi_fw1, wxf_fw1, wxl_fw1, wxo_fw1, whi_fw1, whf_fw1, whl_fw1, who_fw1, bi_fw1, bf_fw1, bl_fw1, bo_fw1,
                   wxi_bw1, wxf_bw1, wxl_bw1, wxo_bw1, whi_bw1, whf_bw1, whl_bw1, who_bw1, bi_bw1, bf_bw1, bl_bw1, bo_bw1,
                   wxi_fw2, wxf_fw2, wxl_fw2, wxo_fw2, whi_fw2, whf_fw2, whl_fw2, who_fw2, bi_fw2, bf_fw2, bl_fw2, bo_fw2,
                   wxi_bw2, wxf_bw2, wxl_bw2, wxo_bw2, whi_bw2, whf_bw2, whl_bw2, who_bw2, bi_bw2, bf_bw2, bl_bw2, bo_bw2,
                   dense_w, dense_b]

        weight_names = ['wxi_fw1', 'wxf_fw1', 'wxl_fw1', 'wxo_fw1', 'whi_fw1', 'whf_fw1', 'whl_fw1', 'who_fw1', 'bi_fw1', 'bf_fw1', 'bl_fw1', 'bo_fw1',
                   'wxi_bw1', 'wxf_bw1', 'wxl_bw1', 'wxo_bw1', 'whi_bw1', 'whf_bw1', 'whl_bw1', 'who_bw1', 'bi_bw1', 'bf_bw1', 'bl_bw1', 'bo_bw1',
                   'wxi_fw2', 'wxf_fw2', 'wxl_fw2', 'wxo_fw2', 'whi_fw2', 'whf_fw2', 'whl_fw2', 'who_fw2', 'bi_fw2', 'bf_fw2', 'bl_fw2', 'bo_fw2',
                   'wxi_bw2', 'wxf_bw2', 'wxl_bw2', 'wxo_bw2', 'whi_bw2', 'whf_bw2', 'whl_bw2', 'who_bw2', 'bi_bw2', 'bf_bw2', 'bl_bw2', 'bo_bw2',
                   'dense_w', 'dense_b']

        weights = {weight_names[w]: weight_vars[w] for w in range(len(weight_vars))}
        weights['units'] = units
        weights['timesteps'] = timesteps
        return weights


# spec = [('units', int32), ('timesteps', int32)]

# spec = [()]
#
# cdef
# int
# units, timesteps
# cdef
# numpy.ndarray[numpy.float32_t, ndim = 2] wxi_fw1, wxf_fw1, wxl_fw1, wxo_fw1, whi_fw1, whf_fw1, whl_fw1, who_fw1
# cdef
# numpy.ndarray[numpy.float32_t, ndim = 1] bi_fw1, bf_fw1, bl_fw1, bo_fw1
# cdef
# numpy.ndarray[numpy.float32_t, ndim = 2] wxi_bw1, wxf_bw1, wxl_bw1, wxo_bw1, whi_bw1, whf_bw1, whl_bw1, who_bw1
# cdef
# numpy.ndarray[numpy.float32_t, ndim = 1] bi_bw1, bf_bw1, bl_bw1, bo_bw1
# cdef
# numpy.ndarray[numpy.float64_t, ndim = 2] input_win
# cdef
# int
# t1
# cdef
# numpy.ndarray[numpy.float64_t, ndim = 1] c_fw, c_bw
# cdef
# numpy.ndarray[numpy.float64_t, ndim = 3] h_fw, h_bw


# @jitclass(spec)
class Rpredict:
    def __init__(self, weight_path='weights.pkl'):
        # with open(weight_path, 'rb') as f:
            # self.units, self.timesteps, self.wxi_fw1, self.wxf_fw1, self.wxl_fw1, self.wxo_fw1, self.whi_fw1, self.whf_fw1, self.whl_fw1, self.who_fw1, self.bi_fw1, self.bf_fw1, self.bl_fw1, self.bo_fw1, self.wxi_bw1, self.wxf_bw1, self.wxl_bw1, self.wxo_bw1, self.whi_bw1, self.whf_bw1, self.whl_bw1, self.who_bw1, self.bi_bw1, self.bf_bw1, self.bl_bw1, self.bo_bw1, self.wxi_fw2, self.wxf_fw2, self.wxl_fw2, self.wxo_fw2, self.whi_fw2, self.whf_fw2, self.whl_fw2, self.who_fw2, self.bi_fw2, self.bf_fw2, self.bl_fw2, self.bo_fw2, self.wxi_bw2, self.wxf_bw2, self.wxl_bw2, self.wxo_bw2, self.whi_bw2, self.whf_bw2, self.whl_bw2, self.who_bw2, self.bi_bw2, self.bf_bw2, self.bl_bw2, self.bo_bw2, self.dense_w, self.dense_b = pickle.load(f)

        weights = pickle.load(open(weight_path, 'rb'))
        self.units = weights['units']
        self.timesteps = weights['timesteps']
        self.wxi_fw1 = weights['wxi_fw1']
        self.wxf_fw1 = weights['wxf_fw1']
        self.wxl_fw1 = weights['wxl_fw1']
        self.wxo_fw1 = weights['wxo_fw1']
        self.whi_fw1 = weights['whi_fw1']
        self.whf_fw1 = weights['whf_fw1']
        self.whl_fw1 = weights['whl_fw1']
        self.who_fw1 = weights['who_fw1']
        self.bi_fw1 = weights['bi_fw1']
        self.bf_fw1 = weights['bf_fw1']
        self.bl_fw1 = weights['bl_fw1']
        self.bo_fw1 = weights['bo_fw1']
        self.wxi_bw1 = weights['wxi_bw1']
        self.wxf_bw1 = weights['wxf_bw1']
        self.wxl_bw1 = weights['wxl_bw1']
        self.wxo_bw1 = weights['wxo_bw1']
        self.whi_bw1 = weights['whi_bw1']
        self.whf_bw1 = weights['whf_bw1']
        self.whl_bw1 = weights['whl_bw1']
        self.who_bw1 = weights['who_bw1']
        self.bi_bw1 = weights['bi_bw1']
        self.bf_bw1 = weights['bf_bw1']
        self.bl_bw1 = weights['bl_bw1']
        self.bo_bw1 = weights['bo_bw1']
        self.wxi_fw2 = weights['wxi_fw2']
        self.wxf_fw2 = weights['wxf_fw2']
        self.wxl_fw2 = weights['wxl_fw2']
        self.wxo_fw2 = weights['wxo_fw2']
        self.whi_fw2 = weights['whi_fw2']
        self.whf_fw2 = weights['whf_fw2']
        self.whl_fw2 = weights['whl_fw2']
        self.who_fw2 = weights['who_fw2']
        self.bi_fw2 = weights['bi_fw2']
        self.bf_fw2 = weights['bf_fw2']
        self.bl_fw2 = weights['bl_fw2']
        self.bo_fw2 = weights['bo_fw2']
        self.wxi_bw2 = weights['wxi_bw2']
        self.wxf_bw2 = weights['wxf_bw2']
        self.wxl_bw2 = weights['wxl_bw2']
        self.wxo_bw2 = weights['wxo_bw2']
        self.whi_bw2 = weights['whi_bw2']
        self.whf_bw2 = weights['whf_bw2']
        self.whl_bw2 = weights['whl_bw2']
        self.who_bw2 = weights['who_bw2']
        self.bi_bw2 = weights['bi_bw2']
        self.bf_bw2 = weights['bf_bw2']
        self.bl_bw2 = weights['bl_bw2']
        self.bo_bw2 = weights['bo_bw2']
        self.dense_w = weights['dense_w']
        self.dense_b = weights['dense_b']


    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def cell(x, c, h, whf, whi, whl, who, wxf, wxi, wxl, wxo, bf, bi, bl, bo):
        new_c = c * Rpredict.sigmoid(h @ whf + x @ wxf + bf) + Rpredict.sigmoid(h @ whi + x @ wxi + bi) * Rpredict.tanh(h @ whl + x @ wxl + bl)
        new_h = Rpredict.tanh(new_c) * Rpredict.sigmoid(h @ who + x @ wxo + bo)
        return new_c, new_h

    def cell_fw1(self, x, c, h):
        new_c = c * Rpredict.sigmoid(h @ self.whf_fw1 + x @ self.wxf_fw1 + self.bf_fw1) + Rpredict.sigmoid(h @ self.whi_fw1 + x @ self.wxi_fw1 + self.bi_fw1) * Rpredict.tanh(h @ self.whl_fw1 + x @ self.wxl_fw1 + self.bl_fw1)
        return new_c, Rpredict.tanh(new_c) * Rpredict.sigmoid(h @ self.who_fw1 + x @ self.wxo_fw1 + self.bo_fw1)

    def cell_bw1(self, x, c, h):
        new_c = c * Rpredict.sigmoid(h @ self.whf_bw1 + x @ self.wxf_bw1 + self.bf_bw1) + Rpredict.sigmoid(h @ self.whi_bw1 + x @ self.wxi_bw1 + self.bi_bw1) * Rpredict.tanh(h @ self.whl_bw1 + x @ self.wxl_bw1 + self.bl_bw1)
        return new_c, Rpredict.tanh(new_c) * Rpredict.sigmoid(h @ self.who_bw1 + x @ self.wxo_bw1 + self.bo_bw1)

    def cell_fw2(self, x, c, h):
        new_c = c * Rpredict.sigmoid(h @ self.whf_fw2 + x @ self.wxf_fw2 + self.bf_fw2) + Rpredict.sigmoid(h @ self.whi_fw2 + x @ self.wxi_fw2 + self.bi_fw2) * Rpredict.tanh(h @ self.whl_fw2 + x @ self.wxl_fw2 + self.bl_fw2)
        return new_c, Rpredict.tanh(new_c) * Rpredict.sigmoid(h @ self.who_fw2 + x @ self.wxo_fw2 + self.bo_fw2)

    def cell_bw2(self, x, c, h):
        new_c = c * Rpredict.sigmoid(h @ self.whf_bw2 + x @ self.wxf_bw2 + self.bf_bw2) + Rpredict.sigmoid(h @ self.whi_bw2 + x @ self.wxi_bw2 + self.bi_bw2) * Rpredict.tanh(h @ self.whl_bw2 + x @ self.wxl_bw2 + self.bl_bw2)
        return new_c, Rpredict.tanh(new_c) * Rpredict.sigmoid(h @ self.who_bw2 + x @ self.wxo_bw2 + self.bo_bw2)

    @staticmethod
    def dense(x, w, b):
        return Rpredict.sigmoid(w @ x + b)

    def predict(self, win):
        c_fw = np.zeros(self.units)
        c_bw = np.zeros(self.units)
        h_fw = np.zeros((self.units, self.timesteps, 2))  # layers=2
        h_bw = np.zeros((self.units, self.timesteps, 2))

        for t_fw, t_bw in zip(range(self.timesteps), range(self.timesteps - 1, -1, -1)):
            c_fw, h_fw[:, t_fw, 0] = self.cell_fw1(win[:, t_fw], c_fw, h_fw[:, t_fw - 1, 0])
            c_bw, h_bw[:, t_bw, 0] = self.cell_bw1(win[:, t_bw], c_bw, h_bw[:, (t_bw + 1) % self.timesteps, 0])

        win = np.concatenate((h_fw[:, :, 0], h_bw[:, :, 0]), axis=0)

        c_fw.fill(0)
        c_bw.fill(0)

        for t_fw, t_bw in zip(range(self.timesteps), range(self.timesteps - 1, -1, -1)):
            c_fw, h_fw[:, t_fw, 1] = self.cell_fw2(win[:, t_fw], c_fw, h_fw[:, t_fw - 1, 1])
            c_bw, h_bw[:, t_bw, 1] = self.cell_bw2(win[:, t_bw], c_bw, h_bw[:, (t_bw + 1) % self.timesteps, 1])

        win = np.concatenate((h_fw[:, :, 1], h_bw[:, :, 1]), axis=0)
        pred_win = Rpredict.dense(win, self.dense_w, self.dense_b)
        return pred_win.reshape(pred_win.shape[1])


class R_plot_training(tf.keras.callbacks.Callback):
    def __init__(self):
        pass

    # def on_train_batch_end(self, epoch, logs={}):
    def on_epoch_end(self, epoch, logs={}):
        fol = 'model_off_ep40_moniTrain'
        nsamples = 3
        # fig, axs = plt.subplots(1, nsamples)
        fig = plt.figure(figsize=(15, 15))
        for s in range(nsamples):
            # ax = fig.add_subplot(nsamples, 1, s + 1)

            sample = logs['inputs'][s].flatten()
            sample_gnt = logs['y_true'][s].flatten()
            sample_prd = logs['pred'][s].flatten()
            times = np.arange(len(sample))

            # ax.plot(times, sample)
            # ax.scatter(x=times[sample_gnt.astype(bool)], y=sample[sample_gnt.astype(bool)], s=80, facecolors='none',
                       # edgecolors='black')
            # ax.scatter(x=times[sample_prd > 0.5], y=sample[sample_prd > 0.5], color='red', marker='x')
            # ax.set_title('Epoch-{} Batch-last Sample-{}'.format(epoch, s))

            sample_prd_thres = sample_prd > 0.5
            if np.any(sample_prd_thres):
                precision = np.sum(sample_gnt & sample_prd_thres) / np.sum(sample_prd_thres)
            else:
                precision = 0
            accuracy = (np.sum(sample_gnt & sample_prd_thres) + np.sum(
                (sample_gnt == 0) & (sample_prd_thres == 0))) / sample_prd_thres.size
            print('')
            print('Accuracy: {:0.2f} | Precision: {:0.2f}'.format(accuracy, precision))
        # plt.tight_layout()
        # fig.savefig('Models/' + fol + '/Training Analysis/' + 'E-{:03d}_B-last_S-first{}'.format(epoch, nsamples))
        # plt.close(fig)
