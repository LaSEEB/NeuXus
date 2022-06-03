import numpy as np
import pickle
from numba import jit
import time


class Rpredict:
    def __init__(self, weight_path=r'C:\Users\guta_\OneDrive\Neuro\Online_correction\models\model6\weights.pkl'):
        self.weights = pickle.load(open(weight_path, 'rb'))
        self.ht = np.zeros((self.weights['t'], self.weights['u']), dtype=np.float32)
        self.c = np.zeros((1, self.weights['u']), dtype=np.float32)

        # Dummy call for numba
        dummy = np.zeros((self.weights['t'], 1), dtype=np.float32)
        t1 = time.perf_counter()
        self.predict(dummy)
        print('detection_time: ', time.perf_counter() - t1)

    def predict(self, xt):
        return self._predict(xt, self.ht, self.c, **self.weights)

    @staticmethod
    @jit(nopython=True)
    def _predict(xt, ht, c, u, t, whf1f, wxf1f, bf1f, whi1f, wxi1f, bi1f, whl1f, wxl1f, bl1f, who1f, wxo1f, bo1f, whf1b, wxf1b, bf1b, whi1b, wxi1b, bi1b, whl1b, wxl1b, bl1b, who1b, wxo1b, bo1b, whf2f, wxf2f, bf2f, whi2f, wxi2f, bi2f, whl2f, wxl2f, bl2f, who2f, wxo2f, bo2f, whf2b, wxf2b, bf2b, whi2b, wxi2b, bi2b, whl2b, wxl2b, bl2b, who2b, wxo2b, bo2b, wd, bd):

        def tanh(a):
            return np.tanh(a)

        def sig(a):
            return 1 / (1 + np.exp(-a))

        def cell(x, h, c, wh1, wx1, b1, wh2, wx2, b2, wh3, wx3, b3, wh4, wx4, b4):
            new_c = c * sig(h @ wh1 + x @ wx1 + b1) + sig(h @ wh2 + x @ wx2 + b2) * tanh(h @ wh3 + x @ wx3 + b3)
            new_h = tanh(new_c) * sig(h @ wh4 + x @ wx4 + b4)
            return new_c, new_h

        def LSTMf(xt, ht, c, t, whf, wxf, bf, whi, wxi, bi, whl, wxl, bl, who, wxo, bo):
            h = ht[t-1:t]
            for i in range(t):
                c, h = cell(xt[i:i + 1], h, c, whf, wxf, bf, whi, wxi, bi, whl, wxl, bl, who, wxo, bo)
                ht[i] = h
            return ht

        def LSTMb(xt, ht, c, t, whf, wxf, bf, whi, wxi, bi, whl, wxl, bl, who, wxo, bo):
            h = ht[0:1]
            for i in range(t-1,-1,-1):
                c, h = cell(xt[i:i + 1], h, c, whf, wxf, bf, whi, wxi, bi, whl, wxl, bl, who, wxo, bo)
                ht[i] = h
            return ht

        def dense(xt, wd, bd):
            return sig(xt @ wd + bd)

        # LSTM-bi 1
        hf = LSTMf(xt, ht.copy(), c, t, whf1f, wxf1f, bf1f, whi1f, wxi1f, bi1f, whl1f, wxl1f, bl1f, who1f, wxo1f, bo1f)
        hb = LSTMb(xt, ht.copy(), c, t, whf1b, wxf1b, bf1b, whi1b, wxi1b, bi1b, whl1b, wxl1b, bl1b, who1b, wxo1b, bo1b)
        xt = np.concatenate((hf, hb), axis=1)
        # LSTM-bi 2
        hf = LSTMf(xt, ht.copy(), c, t, whf2f, wxf2f, bf2f, whi2f, wxi2f, bi2f, whl2f, wxl2f, bl2f, who2f, wxo2f, bo2f)
        hb = LSTMb(xt, ht.copy(), c, t, whf2b, wxf2b, bf2b, whi2b, wxi2b, bi2b, whl2b, wxl2b, bl2b, who2b, wxo2b, bo2b)
        xt = np.concatenate((hf, hb), axis=1)
        # DENSE
        yt = dense(xt, wd, bd)
        return yt[:, 0]
