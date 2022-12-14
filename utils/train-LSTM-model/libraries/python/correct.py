import numpy as np
import pandas as pd
from collections import deque
from wfdb.processing import normalize_bound
from bisect import bisect_left
import pickle
from numba import jit
from scipy.signal import butter, filtfilt


class GA:
    def __init__(self, input_port, marker_input_port, start_marker=None, min_wins=4, max_wins=10, tr=1.260, fs=5000):
        self.channels = input_port.columns
        self.nchans = len(self.channels)
        self.min_wins = min_wins
        self.max_wins = max_wins
        self.npnts = int(tr * fs)                                                   # This should be already an integer value!! So int() just affects the type, not the value
        self.temp_wins = deque(np.zeros((self.max_wins + 1, self.npnts, self.nchans)), maxlen=self.max_wins + 1)
        self.template = np.zeros((self.npnts, self.nchans))
        self.lim1 = 0
        self.wcount = 0
        self.start_marker = start_marker
        self.find_start_marker = (self.start_marker is not None) and (marker_input_port is not None)    # True      False
        self.found_start_marker = False                                                                 # False     False
        self.found_start_point = not self.find_start_marker                                             # False     True
        self.building_start = True
        self.building_start_time = None
        self.subtracting_start = True
        self.subtracting_start_time = None

    def update(self, chunks, markers):
        chunks_output = []
        markers_output = []
        # Find the start time in the marker stream
        if self.find_start_marker:
            for marker in markers:
                marker_values = marker.select_dtypes(include=['object']).values
                if self.start_marker in marker_values:
                    self.building_start_time = marker.index[(marker_values == self.start_marker)[:, 0]][0]
                    self.find_start_marker = False
                    self.found_start_marker = True

        for chunk in chunks:
            clim1 = 0
            # Find the start time in the data stream
            if not self.found_start_point:
                if self.found_start_marker:
                    clim1 = bisect_left(chunk.index, self.building_start_time)
                    if clim1 < len(chunk):                                      # If the current data times alerady reached or passed the start marker time
                        self.found_start_point = True
            # GA-correct
            if self.found_start_point:
                # Send a marker to mark the start of the GA building
                if self.building_start:
                    if not self.found_start_marker:
                        self.building_start_time = chunk.index[0]
                    markers_output.append(pd.DataFrame(data='Start of GA building', index=[self.building_start_time], columns=['marker']))
                    self.building_start = False

                lchunk = len(chunk)
                lim2 = self.lim1 + lchunk - clim1
                for _ in range(0, (lim2 - 1) // self.npnts):                    # For each time the current chunk completes a window: 0, 1 or >1 (in case it is enormous)
                    clim2 = clim1 + (self.npnts - self.lim1)
                    self.fill(chunk, self.npnts, clim1, clim2)                  # Complete the window
                    self.average(self.npnts)                                    # Update the template
                    chunk, markers_output = self.subtract(chunk, self.npnts, clim1, clim2, markers_output)      # Subtract the template to the current chunk
                    self.temp_wins.append(np.zeros((self.npnts, self.nchans)))  # Start a new window
                    self.wcount += 1
                    self.lim1 = 0
                    lim2 = lchunk - clim2
                    clim1 = clim2

                self.fill(chunk, lim2, clim1, lchunk)                           # Continue filling the window
                self.average(lim2)                                              # Update the template
                chunk, markers_output = self.subtract(chunk, lim2, clim1, lchunk, markers_output)  # Subtract the template to the current chunk
                self.lim1 = lim2
            chunks_output.append(chunk)

        return chunks_output, markers_output

    def fill(self, chunk, lim2, clim1, clim2):
        self.temp_wins[-1][self.lim1:lim2, :] = chunk.iloc[clim1:clim2, :]

    def average(self, lim2):
        self.template[self.lim1:lim2, :] = (self.template[self.lim1:lim2, :] * min(self.wcount, self.max_wins) - self.temp_wins[0][self.lim1:lim2, :] + self.temp_wins[-1][self.lim1:lim2, :]) / min((self.wcount + 1), self.max_wins)

    def subtract(self, chunk, lim2, clim1, clim2, markers_output):
        mat = chunk.to_numpy(copy=True)                                                     # Unfortunately subtracting the dataframe raises an error, so it is converted to numpy
        if self.wcount >= self.min_wins - 1:
            mat[clim1:clim2, :] = mat[clim1:clim2, :] - self.template[self.lim1:lim2, :]    # Subtract the template to the current chunk
            # Send a marker to mark the start of the GA subtraction
            if self.subtracting_start:
                self.subtracting_start_time = chunk.index[clim1]
                markers_output.append(pd.DataFrame(data='Start of GA subtraction', index=[self.subtracting_start_time], columns=['marker']))
                self.subtracting_start = False
                print('self.subtracting_start_time = ', self.subtracting_start_time)
        return pd.DataFrame(data=mat, index=chunk.index, columns=chunk.columns), markers_output


class PA:
    def __init__(self, input_port, marker_input_port, dfs, weights_path, start_marker='Start of GA subtraction', numba=True, stride=50, min_wins=10, max_wins=20, min_hc=0.4, max_hc=1.5, short_sight='both', margin=0.1, thres=0.05, filter_ecg=True):
        self.channels = input_port.columns
        self.nchans = len(self.channels)
        self.ecg_id = [i for i, chan in enumerate(self.channels) if chan.upper() in ['ECG', 'EKG']][0]
        self.stride = stride
        self.min_wins = min_wins
        self.max_wins = max_wins
        self.min_hc = round(min_hc * dfs)
        self.max_hc = round(max_hc * dfs)
        self.margin = round(margin * dfs)
        self.short_sight = short_sight
        self.thres = thres
        self.predictor = PredictRPeaks(weights_path, numba=numba)
        self.win_len = self.predictor.t
        self.detect_win = deque(maxlen=self.win_len)
        self.rpeaks_win = deque([False] * self.win_len, maxlen=self.win_len)
        self.start_marker = start_marker
        self.find_start_marker = (self.start_marker is not None) and (marker_input_port is not None)
        self.found_start_marker = False
        self.found_start_point = not self.find_start_marker
        self.building_start = True
        self.building_start_time = None
        self.subtracting_start = True
        self.subtracting_start_time = None
        self.filled_detect_win = False
        self.detection_start = True
        self.detection_time = None
        self.lim1 = 0
        self.hlim = self.win_len - self.stride - self.margin  # Hold limit. Point in the detection window above which data is held until next detection to be output
        self.reached_hold_limit = False
        self.temp_fix = np.zeros((self.nchans, self.max_hc))
        self.temp = np.zeros((self.nchans, self.max_hc))
        self.weights_fix = np.zeros(self.max_hc, dtype=int)
        self.weights = np.zeros(self.max_hc, dtype=int)
        self.wins_fix = deque(np.zeros((self.max_wins + 1, self.max_hc, self.nchans)), maxlen=max_wins + 1)
        self.wins_fix_len = deque(np.zeros((self.max_wins + 1), dtype=int), maxlen=max_wins + 1)
        self.win_fix_len = 0
        self.hcp = self.max_hc
        self.hc = -1
        self.hcp = self.max_hc
        self.hcp_win = np.ones(self.win_len, dtype=int) * (-1)  # Should be integer already!
        self.part_lims = [part_lim for part_lim in range(0, self.win_len, self.stride)] + [self.win_len]                # e.g. [0 250 500 750 1000]
        self.nparts = len(self.part_lims) - 1                                                                           # e.g. 4
        self.pred_wins = deque(maxlen=self.nparts)
        self.time_win = deque(maxlen=self.win_len)
        self.filter_ecg = filter_ecg
        order = 4
        flim = [0.5, 30]
        self._b, self._a = butter(order, [f/(0.5 * dfs) for f in flim], analog=False, btype='band', output='ba')

    def update(self, chunks, markers):
        chunks_output = []
        markers_output = []
        # Find the start time in the marker stream
        if self.find_start_marker:
            for marker in markers:
                marker_values = marker.select_dtypes(include=['object']).values
                if self.start_marker in marker_values:
                    self.building_start_time = marker.index[(marker_values == self.start_marker)[:, 0]][0]
                    self.find_start_marker = False
                    self.found_start_marker = True

        for chunk in chunks:
            clim1 = 0
            # Find the start time in the data stream
            if not self.found_start_point:
                if self.found_start_marker:
                    clim1 = bisect_left(chunk.index, self.building_start_time)
                    if clim1 < len(chunk):                                      # If the current data times alerady reached or passed the start marker time
                        self.found_start_point = True
            # PA-correct
            if self.found_start_point:
                # Send a marker to mark the start of the GA building
                if self.building_start:
                    if not self.found_start_marker:
                        self.building_start_time = chunk.index[0]
                    markers_output.append(pd.DataFrame(data='Start of PA building', index=[self.building_start_time], columns=['marker']))
                    self.building_start = False

                lchunk = len(chunk)
                lim2 = self.lim1 + lchunk - clim1
                if not self.reached_hold_limit:
                    if lim2 < self.hlim:
                        clim2 = lchunk
                        self.fill(chunk, clim1, clim2)
                        # chunks_output.append(pd.DataFrame(data=[self.detect_win[i] for i in range(self.lim1, lim2)], index=[self.time_win[i] for i in range(self.lim1,lim2)], columns=self.channels))
                        chunks_output.append(chunk)
                        self.lim1 = lim2
                    # Detection window data reached hold limit
                    else:
                        clim2 = clim1 + self.hlim - self.lim1
                        self.fill(chunk, clim1, clim2)          # Fill-up detection window
                        # chunks_output.append(pd.DataFrame(data=[self.detect_win[i] for i in range(self.lim1, self.hlim)], index=[self.time_win[i] for i in range(self.lim1, self.hlim)], columns=self.channels))
                        chunks_output.append(chunk.iloc[0:clim2])
                        self.reached_hold_limit = True
                        self.lim1 = - self.margin
                        lim2 = self.lim1 + lchunk - clim2
                        clim1 = clim2

                if self.reached_hold_limit:
                    for _ in range(0, (lim2 - 1) // self.stride):
                        clim2 = clim1 + (self.stride - self.lim1)
                        self.fill(chunk, clim1, clim2)
                        markers_output = self.detect(markers_output)
                        self.make_template()
                        # segment_out = self.detect_win[self.hlim:self.hlim+self.stride]
                        segment_out = pd.DataFrame(data=[self.detect_win[i] for i in range(self.hlim, self.hlim+self.stride)], index=[self.time_win[i] for i in range(self.hlim, self.hlim+self.stride)], columns=self.channels)
                        segment_out, markers_output = self.subtract(segment_out, markers_output)
                        chunks_output.append(segment_out)
                        clim1 = clim2
                        self.lim1 = 0
                        lim2 = lchunk - clim2

                    self.fill(chunk, clim1, lchunk)
                    self.lim1 = lim2

            else:
                chunks_output.append(chunk)
        return chunks_output, markers_output

    def fill(self, chunk, clim1, clim2):
        extension = chunk.iloc[clim1:clim2]
        extension_len = len(extension)
        self.detect_win.extend(extension.to_numpy(copy=True))
        self.time_win.extend(extension.index.to_numpy(copy=True))
        self.rpeaks_win.extend(np.zeros(extension_len, dtype=bool))

    def detect(self, markers_output):
        # Extract ECG
        # ecg_win = np.asarray(self.detect_win, dtype=np.float32)[:, self.ecg_id:self.ecg_id + 1]
        # ecg_win = np.asarray([self.detect_win[i][self.ecg_id] for i in range(self.win_len)], dtype=np.float32)[:, None]
        ecg_win = np.asarray([timepoints[self.ecg_id] for timepoints in self.detect_win])
        # Filter
        if self.filter_ecg:
            ecg_win = filtfilt(self._b, self._a, ecg_win)
        # Add singleton dimension and cast to float32 (numba predictor expects a [win_len x 1] float32 array)
        ecg_win = np.float32(ecg_win[:, None])
        # Normalize
        norm_win = normalize_bound(ecg_win, lb=-1, ub=1)
        # Estimate R peak probabilities
        pred_win = self.predictor.predict(norm_win)
        # Average overlapping parts in last prediction windows
        self.pred_wins.appendleft(pred_win)
        avg_win = np.zeros(len(pred_win))
        max_height = 0
        for pi in range(self.nparts - 1, -1, -1):
            max_height += 1
            parts = []
            for wi in range(min(len(self.pred_wins), max_height)):  # e.g. for part = 4: stack 1 win; if part = 3: stack 1 win if wins = 1, stack 2 wins if wins = 2;
                parts.append(self.pred_wins[wi][self.part_lims[pi + wi]:self.part_lims[pi + 1 + wi]])
            avg_win[self.part_lims[pi]:self.part_lims[pi + 1]] = np.asarray(parts).mean(axis=0)
        # Threshold
        peak_ids = np.where(avg_win > self.thres)[0]
        # Snap detections to local maxima
        snap_ids = WFDBPeaks.correct_peaks(sig=ecg_win, peak_inds=peak_ids, search_radius=5, smooth_window_size=20, peak_dir='up')  # e.g. array([39,39,39,39,39, 101,101, 142,142,142,142,142, 180])
        # Filter maxima (consider just those w/ 5 snaps or more)
        vals, counts = np.unique(snap_ids, return_counts=True)
        filt_ids = np.extract(counts >= 5, vals)
        # Remove close maxima with lower prediction score
        dist = 0
        if len(filt_ids) > 0:
            close_ids = [filt_ids[0]]
            dist_scores = np.ones(len(filt_ids), dtype=bool)
            for i in range(1, len(filt_ids)):
                dist = dist + filt_ids[i] - filt_ids[i - 1]
                if dist < self.min_hc:
                    close_ids.append(filt_ids[i])
                if dist >= self.min_hc:
                    max_id = np.argmax(avg_win[close_ids])
                    for j in range(len(close_ids)):
                        if j != max_id:
                            dist_scores[i-len(close_ids) + j] = False
                    close_ids = [filt_ids[i]]
                    dist = 0
                elif i == len(filt_ids)-1:
                    max_id = np.argmax(avg_win[close_ids])
                    for j in range(len(close_ids)):
                        if j != max_id:
                            dist_scores[i-len(close_ids) + 1 + j] = False
            filt_ids = filt_ids[dist_scores]
        # Update deque with predictions
        self.rpeaks_win = deque([False] * self.win_len, maxlen=self.win_len)
        for fi in filt_ids:
            self.rpeaks_win[fi] = True
            if fi < self.stride:
                markers_output.append(pd.DataFrame(data='R peak fixed', index=[self.time_win[fi]], columns=['marker']))
            elif fi < self.win_len - self.margin:
                markers_output.append(pd.DataFrame(data='R peak', index=[self.time_win[fi]], columns=['marker']))
            else:
                markers_output.append(pd.DataFrame(data='R peak marginalized', index=[self.time_win[fi]], columns=['marker']))

        # Send a marker to mark the start of the R peak detection
        self.detection_time = self.time_win[-1]
        if self.detection_start:
            markers_output.append(pd.DataFrame(data='Start of R peak detection', index=[self.detection_time], columns=['marker']))
            self.detection_start = False
        markers_output.append(pd.DataFrame(data='R peak detection', index=[self.detection_time], columns=['marker']))
        markers_output.append(pd.DataFrame(data='Hold limit', index=[self.time_win[self.hlim]], columns=['marker']))
        markers_output.append(pd.DataFrame(data='Margin', index=[self.time_win[self.hlim + self.stride]], columns=['marker']))

        return markers_output

    def make_template(self):
        self.hcp_win.fill(self.max_hc)
        for i in range(self.stride):
            self.hcp += 1
            if self.rpeaks_win[i]:
                self.wins_fix.append(np.zeros((self.max_hc, self.nchans)))  # self.tempwins[-1][self.lim1:lim2, :] = chunk[clim1:clim2, :]
                self.wins_fix_len.append(0)
                self.hcp = 0
                self.hc += 1  # Number of complete unmodifiable heart cycles
                weights_fix_last = self.weights_fix.copy()
                weights_clipped = self.weights_fix.copy()
                weights_clipped[:self.wins_fix_len[0]] -= 1
                weights_clipped[weights_clipped < 1] = 1
                self.temp_fix = (self.temp_fix * weights_fix_last - np.transpose(self.wins_fix[0])) / weights_clipped
                self.weights_fix[:self.wins_fix_len[0]] -= 1

            if self.hcp < self.max_hc:
                self.weights_fix[self.hcp] = self.weights_fix[self.hcp] + 1
                self.temp_fix[:, self.hcp] = (self.temp_fix[:, self.hcp] * (self.weights_fix[self.hcp] - 1) + self.detect_win[i]) / self.weights_fix[self.hcp]
                self.wins_fix[-1][self.hcp, :] = self.detect_win[i]
                self.hcp_win[i] = self.hcp
                self.wins_fix_len[-1] += 1

        self.weights = self.weights_fix.copy()
        self.temp = self.temp_fix.copy()

        for i in range(self.stride, self.win_len - self.margin):
            self.hcp += 1
            if self.rpeaks_win[i]:
                self.hcp = 0

            if self.hcp < self.max_hc:
                self.weights[self.hcp] = self.weights[self.hcp] + 1
                self.temp[:, self.hcp] = (self.temp[:, self.hcp] * (self.weights[self.hcp] - 1) + self.detect_win[i]) / self.weights[self.hcp]
                self.hcp_win[i] = self.hcp

        self.hcp = self.hcp_win[self.stride-1]

    def subtract(self, segment, markers_output):
        segment_hc_labels = self.hcp_win[self.hlim:self.hlim + self.stride]
        mask_len = segment_hc_labels < self.max_hc
        mask_wei = self.weights[segment_hc_labels * mask_len] > self.min_wins
        mask = mask_len * mask_wei
        if any(mask):
            segment.iloc[mask, :self.ecg_id] -= np.transpose(self.temp[:self.ecg_id, segment_hc_labels[mask]])
            if self.subtracting_start:
                self.subtracting_start_time = segment.index[np.argmax(mask)]
                markers_output.append(pd.DataFrame(data='Start of PA subtraction', index=[self.subtracting_start_time], columns=['marker']))
                self.subtracting_start = False
        return segment, markers_output


class PredictRPeaks:
    def __init__(self, weight_path, numba=True):
        self.weights = pickle.load(open(weight_path, 'rb'))
        self.ht = np.zeros((self.weights['t'], self.weights['u']), dtype=np.float32)
        self.c = np.zeros((1, self.weights['u']), dtype=np.float32)
        self.t = self.weights['t']
        self.numba = numba

        if numba:
            dummy = np.zeros((self.t, 1), dtype=np.float32)
            # t1 = time.perf_counter()
            self.predict(dummy)
            # print('Numba compilation time: ', time.perf_counter() - t1)


    def predict(self, xt):
        if self.numba:
            return self._predict_numba(xt, self.ht, self.c, **self.weights)
        else:
            return self._predict(xt, self.ht, self.c, **self.weights)

    # Predict using Numba
    @staticmethod
    @jit(nopython=True)
    def _predict_numba(xt, ht, c, u, t, whf1f, wxf1f, bf1f, whi1f, wxi1f, bi1f, whl1f, wxl1f, bl1f, who1f, wxo1f, bo1f, whf1b, wxf1b, bf1b, whi1b, wxi1b, bi1b, whl1b, wxl1b, bl1b, who1b, wxo1b, bo1b, whf2f, wxf2f, bf2f, whi2f, wxi2f, bi2f, whl2f, wxl2f, bl2f, who2f, wxo2f, bo2f, whf2b, wxf2b, bf2b, whi2b, wxi2b, bi2b, whl2b, wxl2b, bl2b, who2b, wxo2b, bo2b, wd, bd):

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
            for i in range(t-1, -1, -1):
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

    # Predict without using Numba
    @staticmethod
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


# Gustavo's note:
# The R detection uses two functions from the library "wfdb", found in: wfdb.processing.peaks
# I have copied them here because I have made two changes:
# (1) I have commented the smoothing step in correct_peaks (I already specified an option to bandpass the ecg)
# (2) I changed shift_peaks because I found it to be incorrect.
# Until the wfdb library accomodates theses changes, I am using them from here:
class WFDBPeaks:

    @staticmethod
    def correct_peaks(sig, peak_inds, search_radius, smooth_window_size,
                      peak_dir='compare'):
        """
        Adjust a set of detected peaks to coincide with local signal maxima,
        and

        Parameters
        ----------
        sig : numpy array
            The 1d signal array
        peak_inds : np array
            Array of the original peak indices
        max_gap : int
            The radius within which the original peaks may be shifted.
        smooth_window_size : int
            The window size of the moving average filter applied on the
            signal. Peak distance is calculated on the difference between
            the original and smoothed signal.
        peak_dir : str, optional
            The expected peak direction: 'up' or 'down', 'both', or
            'compare'.

            - If 'up', the peaks will be shifted to local maxima
            - If 'down', the peaks will be shifted to local minima
            - If 'both', the peaks will be shifted to local maxima of the
              rectified signal
            - If 'compare', the function will try both 'up' and 'down'
              options, and choose the direction that gives the largest mean
              distance from the smoothed signal.

        Returns
        -------
        corrected_peak_inds : numpy array
            Array of the corrected peak indices


        Examples
        --------

        """
        sig_len = sig.shape[0]
        n_peaks = len(peak_inds)

        # Subtract the smoothed signal from the original
        # sig = sig - smooth(sig=sig, window_size=smooth_window_size)

        # Shift peaks to local maxima
        if peak_dir == 'up':
            shifted_peak_inds = WFDBPeaks.shift_peaks(sig=sig,
                                            peak_inds=peak_inds,
                                            search_radius=search_radius,
                                            peak_up=True)
        elif peak_dir == 'down':
            shifted_peak_inds = WFDBPeaks.shift_peaks(sig=sig,
                                            peak_inds=peak_inds,
                                            search_radius=search_radius,
                                            peak_up=False)
        elif peak_dir == 'both':
            shifted_peak_inds = WFDBPeaks.shift_peaks(sig=np.abs(sig),
                                            peak_inds=peak_inds,
                                            search_radius=search_radius,
                                            peak_up=True)
        else:
            shifted_peak_inds_up = WFDBPeaks.shift_peaks(sig=sig,
                                               peak_inds=peak_inds,
                                               search_radius=search_radius,
                                               peak_up=True)
            shifted_peak_inds_down = WFDBPeaks.shift_peaks(sig=sig,
                                                 peak_inds=peak_inds,
                                                 search_radius=search_radius,
                                                 peak_up=False)

            # Choose the direction with the biggest deviation
            up_dist = np.mean(np.abs(sig[shifted_peak_inds_up]))
            down_dist = np.mean(np.abs(sig[shifted_peak_inds_down]))

            if up_dist >= down_dist:
                shifted_peak_inds = shifted_peak_inds_up
            else:
                shifted_peak_inds = shifted_peak_inds_down

        return shifted_peak_inds

    @staticmethod
    def shift_peaks(sig, peak_inds, search_radius, peak_up):
        """
        Helper function for correct_peaks. Return the shifted peaks to local
        maxima or minima within a radius.

        peak_up : bool
            Whether the expected peak direction is up
        """
        sig_len = sig.shape[0]
        n_peaks = len(peak_inds)
        # The indices to shift each peak ind by
        shift_inds = np.zeros(n_peaks, dtype='int')

        # Iterate through peaks
        for i in range(n_peaks):
            ind = peak_inds[i]
            # Gustavo: (why not go to the end?!)
            # local_sig = sig[max(0, ind - search_radius):min(ind + search_radius, sig_len-1)]
            local_sig = sig[max(0, ind - search_radius):min(ind + search_radius + 1, sig_len)]

            if peak_up:
                shift_inds[i] = np.argmax(local_sig)
            else:
                shift_inds[i] = np.argmin(local_sig)

        # May have to adjust early values
        for i in range(n_peaks):
            ind = peak_inds[i]
            if ind >= search_radius:
                break
            # Gustavo: just wrong
            # shift_inds[i] -= search_radius - ind
            shift_inds[i] += search_radius - ind

        shifted_peak_inds = peak_inds + shift_inds - search_radius

        return shifted_peak_inds