import numpy as np
from neuxus.node import Node
from collections import deque
import pandas as pd
from bisect import bisect_left
from neuxus.nodes.temporary_peaks import correct_peaks
from wfdb.processing import normalize_bound
from neuxus.nodes.detect import Rpredict
from neuxus.chunks import Port
import time

import pickle

class GA(Node):
    def __init__(self, input_port, minwins=4, maxwins=10, nchans=32, fs=5000, tr=1.260, start_marker=None, marker_input_port=None):
        Node.__init__(self, input_port)
        self.output.set_parameters(
            data_type=self.input.data_type,
            channels=self.input.channels,
            sampling_frequency=self.input.sampling_frequency,
            meta=self.input.meta,
            epoching_frequency=self.input.epoching_frequency,
        )

        self.marker_output = Port()
        self.marker_output.set_parameters(
            data_type='marker',
            channels=['marker'],
            sampling_frequency=0,
            meta=''
        )
        self.ga_corrected = False

        self.start = (start_marker is None) or (marker_input_port is None)
        self.marker_input_port = marker_input_port
        self.start_marker = start_marker

        self.nchans = nchans
        self.minwins = minwins
        self.maxwins = maxwins
        self.npnts = int(tr * fs)  # This should be already a .0 number!!, so int just affects the type, not the value
        self.tempwins = deque(np.zeros((self.maxwins + 1, self.npnts, self.nchans)), maxlen=self.maxwins + 1)
        self.template = np.zeros((self.npnts, self.nchans))
        self.lim1 = 0
        self.wcount = 0

    def update(self):
        for chunk in self.marker_input_port:
            if not self.start:
                values = chunk.select_dtypes(include=['object']).values
                print('values = ', values)
                print('self.start_marker = ', self.start_marker)
                self.start = self.start_marker in values
                print('self.start = ', self.start)

        for chunk in self.input:
            if self.start:
                index = chunk.index
                chunk = chunk.to_numpy(copy=True)
                lchunk = len(chunk)
                lim2 = self.lim1 + lchunk
                clim1 = 0
                for i in range(0, (lim2 - 1) // self.npnts):
                    clim2 = clim1 + (self.npnts - self.lim1)
                    self.fill(chunk, self.npnts, clim1, clim2)
                    self.average(self.npnts)
                    if self.wcount >= self.minwins - 1:
                        chunk = self.subtract(chunk, self.npnts, clim1, clim2)
                    self.tempwins.append(np.zeros((self.npnts, self.nchans)))
                    self.wcount += 1
                    self.lim1 = 0
                    lim2 = lchunk - clim2
                    clim1 = clim2

                self.fill(chunk, lim2, clim1, lchunk)
                self.average(lim2)
                if self.wcount >= self.minwins - 1:
                    chunk = self.subtract(chunk, lim2, clim1, lchunk)
                    if not self.ga_corrected:
                        self.ga_corrected = True
                        self.marker_output.set(['Start of GA correction'], [index[clim1]])
                self.lim1 = lim2
                self.output.set(chunk, index, self.input.channels)
            else:
                index = chunk.index
                self.output.set(chunk, index, self.input.channels)

    def fill(self, chunk, lim2, clim1, clim2):
        self.tempwins[-1][self.lim1:lim2, :] = chunk[clim1:clim2, :]

    def average(self, lim2):
        self.template[self.lim1:lim2, :] = (self.template[self.lim1:lim2, :] * min(self.wcount, self.maxwins) - self.tempwins[0][self.lim1:lim2, :] + self.tempwins[-1][self.lim1:lim2, :]) / min((self.wcount + 1), self.maxwins)

    def subtract(self, chunk, lim2, clim1, clim2):
        chunk[clim1:clim2, :] = chunk[clim1:clim2, :] - self.template[self.lim1:lim2, :]
        return chunk


class PA(Node):
    def __init__(self, input_port, weight_path, win_len, stride, start_marker=None, marker_input_port=None, min_weight=10, max_wins=15, min_hc=0.4, max_hc=1.5, short_sight='both', min_foresight=0.1, thres=0.05):
        Node.__init__(self, input_port)
        fs = self.input.sampling_frequency
        self.fs = fs
        self.output.set_parameters(
            data_type=self.input.data_type,
            channels=self.input.channels,
            sampling_frequency=fs,
            meta=self.input.meta,
            epoching_frequency=self.input.epoching_frequency
        )
        self.marker_output_pa = Port()
        self.marker_output_pa.set_parameters(
            data_type='marker',
            channels=['marker'],
            sampling_frequency=0,
            meta=''
        )
        self.marker_output_r = Port()
        self.marker_output_r.set_parameters(
            data_type='marker',
            channels=['marker'],
            sampling_frequency=0,
            meta=''
        )
        self.nchans = len(self.input.channels)
        self.ecg_chn = self.input.channels.index('ECG')
        self.filled = False

        self.start = (start_marker is None) or (marker_input_port is None)
        self.marker_input_port = marker_input_port
        self.start_marker = start_marker

        self.lim1 = 0
        self.win_len = win_len
        self.new = 0
        self.stride = stride

        self.eegwin = deque(maxlen=win_len)
        self.twin = deque(maxlen=win_len)
        self.chunk_keep = pd.DataFrame()
        self.chunk_keep_hcp = np.empty(0, dtype=int)

        self.max_hc_len = round(max_hc * fs)
        self.hcp_win = np.ones(win_len, dtype=int) * (-1)  # Should be integer already!
        self.hc = -1
        self.hcp = self.max_hc_len
        self.temp_fix = np.zeros((self.nchans, self.max_hc_len))
        self.temp = np.zeros((self.nchans, self.max_hc_len))
        self.weights_fix = np.zeros(self.max_hc_len, dtype=int)
        self.weights = np.zeros(self.max_hc_len, dtype=int)
        self.min_weight = min_weight
        self.wins_fix = deque(np.zeros((max_wins + 1, self.max_hc_len, self.nchans)), maxlen=max_wins + 1)
        self.wins_fix_len = deque(np.zeros((max_wins + 1), dtype=int), maxlen=max_wins + 1)

        self.predictor = Rpredict(weight_path)
        self.part_lims = [part_lim for part_lim in range(0, self.win_len, self.stride)] + [self.win_len]                # e.g. [0 250 500 750 1000]
        self.nparts = len(self.part_lims) - 1                                                                           # e.g. 4
        self.pred_wins = deque(maxlen=self.nparts)
        self.thres = thres
        self.min_hc_len = min_hc * fs

        self.short_sight = short_sight
        self.min_foresight_len = int(min_foresight * fs)

        self.pa_corrected = False

        self.debug_save = False

        # self.tol = 1 / fs / 2  # Half period

    # def find_start_of_ga_corrected(self, chunk):
    #     clim1 = 0
    #     if not self.ga_corrected_de_facto:
    #         if self.input.ga_corrected:
    #             if bisect_left(chunk.index, self.input.tlim1 - self.tol) < len(chunk):
    #                 self.ga_corrected_de_facto = True
    #                 clim1 = bisect_left(chunk.index, tlim1_ga)
    #     return clim1

    def find_start_of_ga_corrected_and_trim(self, chunk):
        values = chunk.select_dtypes(include=['object']).values
        self.start = self.start_marker in values
        if self.start:
            tlim1_ga = chunk.index[chunk[0]==self.start_marker][0]
            for i in range(len(self.input._data)):
                self.input._data[i] = self.input._data[i][tlim1_ga:]  # REMOVE IF EMPTY
            if self.input._data[0].empty:
                self.input._data = []
            # print('self.input._data = ', self.input._data)


    def update(self):
        for chunk in self.marker_input_port:
            if not self.start:
                self.find_start_of_ga_corrected_and_trim(chunk)

        for chunk in self.input:
            # clim1 = self.find_start_of_ga_corrected(chunk)
            if self.start:
                clim1 = 0
                lchunk = len(chunk)
                detected = False
                if not self.filled:
                    if (lchunk - clim1) < (self.win_len - self.lim1):
                        self.fill(chunk, clim1, lchunk)
                        chunk_out = chunk
                        # print('1) len(chunk_out) = ', len(chunk_out))
                        self.output.set_from_df(chunk_out)
                        self.lim1 = self.lim1 + lchunk - clim1
                    else:
                        clim2 = clim1 + (self.win_len - self.lim1)  # clim1 WILL BE ALMOST ALWAYS = 0 (BUT IN THE FIRST GA-CORRECTED CHUNK IT MIGHT NOT)
                        self.fill(chunk, clim1, clim2)
                        self.fill_buffers(chunk, 0, clim2)  # For the case where first chunk is huge: include all in buffer (to send out), but include only part after GA in wins
                        # print('clim2 = ', clim2)
                        # print('chunk.index = ', chunk.index)
                        # print('chunk.index[clim2] = ', chunk.index[clim2])
                        self.detect(chunk.index[clim2-1])
                        self.make_template()
                        self.label_chunk_keep()
                        clim1 = clim2
                        self.filled = True
                        detected = True

                if self.filled:
                    for i in range(0, ((lchunk - clim1) + self.new) // self.stride):
                        clim2 = clim1 + self.stride - self.new
                        self.fill(chunk, clim1, clim2)
                        self.fill_buffers(chunk, clim1, clim2)
                        self.detect(chunk.index[clim2-1])
                        self.make_template()
                        self.label_chunk_keep()
                        clim1 = clim2
                        self.new = 0
                        detected = True

                    if detected:
                        klim = len(self.chunk_keep) - self.short_sight_len
                        if klim > 0:
                            # klim = max(len(self.chunk_keep) - self.short_sight_len, 0)
                            chunk_out = self.chunk_keep.iloc[:klim].copy()
                            chunk_hcp = self.chunk_keep_hcp[:klim].copy()
                            self.chunk_keep = self.chunk_keep.iloc[klim:]
                            self.chunk_keep_hcp = self.chunk_keep_hcp[klim:]
                            chunk_out = self.subtract(chunk_out, chunk_hcp)
                            # print('2) len(chunk_out) = ', len(chunk_out))
                            self.output.set_from_df(chunk_out)

                    else:
                        pass
                        # chunk_out = self.chunk_keep.iloc[0:0]

                    self.fill(chunk, clim1, lchunk)
                    self.fill_buffers(chunk, clim1, lchunk)
                    self.new = self.new + lchunk - clim1  # new points not yet used for detection
            else:
                chunk_out = chunk
                # print('3) len(chunk_out) = ', len(chunk_out))
                self.output.set_from_df(chunk_out)

            # self.output.set_from_df(chunk_out)

    def subtract(self, chunk, chunk_hcp):
        mat = np.asarray(chunk)
        mask_len = chunk_hcp < self.max_hc_len
        mask_wei = self.weights[chunk_hcp * mask_len] > self.min_weight
        mask = mask_len * mask_wei
        mat[mask] -= np.transpose(self.temp[:, chunk_hcp[mask]])
        chunk[:] = mat
        if any(mask):
            if not self.pa_corrected:
                self.pa_corrected = True
                self.marker_output_pa.set(['Start of PA correction'], [chunk.index[np.argmax(mask)]])
        return chunk

    def label_chunk_keep(self):
        wlim = max(self.win_len - len(self.chunk_keep_hcp), 0)
        copy = self.hcp_win[wlim:]
        clim = len(self.chunk_keep_hcp) - len(copy)
        self.chunk_keep_hcp[clim:] = copy

    def make_template(self):
        self.hcp_win.fill(self.max_hc_len)
        for i in range(self.stride):
            self.hcp += 1
            if self.rwin[i]:
                self.r_found = True
                self.wins_fix.append(np.zeros((self.max_hc_len, self.nchans)))  # self.tempwins[-1][self.lim1:lim2, :] = chunk[clim1:clim2, :]
                self.wins_fix_len.append(0)
                self.hcp = 0
                self.hc += 1  # Number of complete unmodifiable heart cycles
                weights_fix_last = self.weights_fix.copy()
                weights_clipped = self.weights_fix.copy()
                weights_clipped[:self.wins_fix_len[0]] -= 1
                weights_clipped[weights_clipped < 1] = 1
                self.temp_fix = (self.temp_fix * weights_fix_last - np.transpose(self.wins_fix[0])) / weights_clipped
                self.weights_fix[:self.wins_fix_len[0]] -= 1

            if self.hcp < self.max_hc_len:
                self.weights_fix[self.hcp] = self.weights_fix[self.hcp] + 1
                self.temp_fix[:, self.hcp] = (self.temp_fix[:, self.hcp] * (self.weights_fix[self.hcp] - 1) + self.eegwin[i]) / self.weights_fix[self.hcp]
                self.wins_fix[-1][self.hcp, :] = self.eegwin[i]
                self.hcp_win[i] = self.hcp
                self.wins_fix_len[-1] += 1

        self.weights = self.weights_fix.copy()
        self.temp = self.temp_fix.copy()

        for i in range(self.stride, self.win_len):
            self.hcp += 1

            if self.rwin[i]:
                self.hcp = 0

            if self.hcp < self.max_hc_len:
                self.weights[self.hcp] = self.weights[self.hcp] + 1
                self.temp[:, self.hcp] = (self.temp[:, self.hcp] * (self.weights[self.hcp] - 1) + self.eegwin[i]) / self.weights[self.hcp]
                self.hcp_win[i] = self.hcp

        self.hcp = self.hcp_win[self.stride-1]

    def detect(self, last_time):
        # print('last_time = ', last_time)
        ecg_win = np.asarray(self.eegwin, dtype=np.float32)[:, self.ecg_chn:self.ecg_chn+1]  # ecg_win = np.asarray([self.eegwin[i][self.ecg_chn] for i in range(len(self.eegwin))])
        norm_win = normalize_bound(ecg_win, lb=-1, ub=1)

        t1 = time.perf_counter()
        pred_win = self.predictor.predict(norm_win)
        print('detection_time: ', time.perf_counter() - t1)

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

        snap_ids = correct_peaks(sig=ecg_win,
                                peak_inds=peak_ids,
                                search_radius=5,
                                smooth_window_size=20,
                                peak_dir='up')  # e.g. array([39,39,39,39,39, 101,101, 142,142,142,142,142, 180])

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
                if dist < self.min_hc_len:
                    close_ids.append(filt_ids[i])
                if dist >= self.min_hc_len:
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

        # self.marker_output_r.set(['R'], [chunk.index[np.argmax(mask)]])
        # print('type(filt_ids) = ', type(filt_ids))

        # times_r = last_time - (self.win_len - 1 - filt_ids) / self.fs
        # print(times_r)

        # Update deque with predictions
        self.rwin = deque([False] * self.win_len, maxlen=self.win_len)
        for fi in filt_ids:
            self.rwin[fi] = True
            self.marker_output_r.set(['R'], [last_time - (self.win_len - 1 - fi) / self.fs])


        # Short-sight
        if self.short_sight == 'both':
            self.short_sight_len = self.min_foresight_len

        elif self.short_sight == 'positive':
            short_sighted = int(filt_ids[np.searchsorted(filt_ids, self.min_foresight_id, side='right'):])
            self.short_sight_len = np.append(self.win_len - short_sighted, 0)[0]

        self.detected = True

    def fill(self, chunk, clim1, clim2):
        part = chunk.iloc[clim1:clim2]
        part_len = len(part)
        self.eegwin.extend(part.to_numpy().copy())
        if self.debug_save:
            pickle.dump(self.eegwin, open('eegwin.pkl', 'wb'))
            self.debug_save = True

        # print(self.eegwin)
        # self.short_sighted -= part_len

    def fill_buffers(self, chunk, clim1, clim2):
        part = chunk.iloc[clim1:clim2]
        self.chunk_keep = self.chunk_keep.append(part)
        new_hcp_array = self.create_constant_array(len(part), self.max_hc_len)
        self.chunk_keep_hcp = np.append(self.chunk_keep_hcp, new_hcp_array)

    def find_start_of_ga_corrected(self, chunk):
        clim1 = 0
        if not self.ga_corrected_de_facto:
            if self.input.ga_corrected:
                if bisect_left(chunk.index, self.input.tlim1 - self.tol) < len(chunk):
                    self.ga_corrected_de_facto = True
                    clim1 = bisect_left(chunk.index, tlim1_ga)
        return clim1

    @staticmethod
    def create_constant_array(length, value):
        array = np.empty(length, dtype=int)
        array.fill(value)
        return array


# z = pickle.load(open(r'C:\Users\guta_\OneDrive\Neuro\Online_correction\tests\neuxus_integration\eegwin.pkl', 'rb'))
