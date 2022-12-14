import numpy as np
import scipy.io as sio
from numpy.core.records import fromarrays
from bisect import bisect_left


class tools:

    @staticmethod
    def load(path):
        mat = sio.loadmat(path, struct_as_record=False)
        EEG = mat['EEG'][0, 0]  # EEG = mat['EEGECG'][0, 0]

        for e in range(len(EEG.event[0])):
            EEG.event[0][e].latency[0, 0] -= 1

        # Turn MATLAB's 1-indexing to 0-indexing
        # events = EEG.event[0]
        # types = [events[e].type[0] for e in range(len(events))]
        # latencies = [events[e].latency[0, 0] - 1 for e in range(len(events))]
        # events = fromarrays([np.asarray(types), np.asarray(latencies)], names=["type", "latency"])
        # EEG.event[0] = events
        return EEG

    @staticmethod
    def trim(EEG, event1='S  1', event2='S 12', timeshifts=[0, 0]):
        fs = EEG.srate[0, 0]
        events = EEG.event[0]
        types = [events[e].type[0] for e in range(len(events))]
        latencies = [events[e].latency[0, 0] for e in range(len(events))]
        start_e, start_lat = [(e, latencies[e]) for e in range(events.size) if event1 == types[e]][0]
        end_e, end_lat = [(e, latencies[e]) for e in range(events.size) if event2 == types[e]][-1]  # It was [0]

        start_lat = max(0, start_lat + round(timeshifts[0]*fs))
        end_lat = min(len(EEG.times[0]), end_lat + round(timeshifts[1] * fs))

        data = EEG.data[:, start_lat:end_lat + 1].astype(np.float64)
        times = EEG.times[:, start_lat:end_lat + 1].astype(np.float64)
        times = times - times[0, 0]

        xmin = times[0, 0]
        xmax = times[0, -1]
        pnts = times.size

        latencies = np.array(latencies)
        mask = np.logical_and(latencies >= start_lat, latencies <= end_lat)
        events = events[mask]
        first_lat = start_lat

        # events = [event for event in events if (start_lat <= event.latency[0, 0] <= end_lat)]
        # events = events[start_e:end_e + 1]
        # first_lat = events[0].latency
        for e in range(events.size):
            # events[e].latency = events[e].latency - first_lat + 1
            events[e].latency = events[e].latency - first_lat
            events[e].urevent = e + 1

        EEG.data = data
        EEG.times = times
        EEG.xmin = xmin
        EEG.xmax = xmax
        EEG.pnts = pnts
        EEG.event = np.expand_dims(events, axis=0)
        return EEG

    @staticmethod
    def unpack(EEG):
        data = EEG.data
        times = EEG.times / 1000
        fs = EEG.srate[0, 0]
        chans = [EEG.chanlocs[0, chn].labels[0] for chn in range(len(EEG.chanlocs[0]))]
        events = EEG.event[0]
        types = [events[e].type[0] for e in range(len(events))]
        latencies = [events[e].latency[0, 0] for e in range(len(events))]
        return data, times, fs, chans, types, latencies


    @staticmethod
    def pack(fname, data, times, fs, chans, types, latencies):
        chanlocs = fromarrays([chans], names=["labels"])
        sort_indices = np.argsort(latencies)
        types = [types[i] for i in sort_indices]
        latencies = [latencies[i] for i in sort_indices]
        events = fromarrays([np.asarray(types), np.asarray(latencies)], names=["type", "latency"])

        EEG = dict(EEG=dict(setname='',
                      filename=fname,
                      filepath='',
                      data=data,
                      times=times*1000,
                      nbchan=data.shape[0],
                      pnts=data.shape[1],
                      trials=1,
                      srate=float(fs),
                      xmin=np.squeeze(times)[0],
                      xmax=np.squeeze(times)[-1],
                      chanlocs=chanlocs,
                      event=events,
                      icawinv=[],
                      icasphere=[],
                      icaweights=[],
                      icaact=[]
                      ))

        return EEG

    @staticmethod
    def save(fname, EEG):
        # Turn Python's 0-indexing to MATLAB's 1-indexing
        for e in range(len(EEG['EEG']['event'])):
            EEG['EEG']['event'][e][1] += 1
        lala = 0
        # events = EEG.event[0]
        # types = [events[e].type[0] for e in range(len(events))]
        # latencies = [events[e].latency[0, 0] + 1 for e in range(len(events))]
        # events = fromarrays([np.asarray(types), np.asarray(latencies)], names=["type", "latency"])
        # EEG.event[0] = events

        sio.savemat(fname, EEG, appendmat=False)

    @staticmethod
    def fit_latencies(latencies1, times1, times2):
        # latencies2 = [tools.find_closest_index(times2, times1[int(latencies1[e] - 1)]) + 1 for e in range(len(latencies1))]  # The -1 and +1 are to change between pythons [0 len-1] to matlabs [1 len]
        latencies2 = [tools.find_closest_index(times2, times1[int(latencies1[e])]) for e in range(len(latencies1))]
        return latencies2

    @staticmethod
    def find_closest_index(a, x):
        i = bisect_left(a, x)
        if i >= len(a):
            i = len(a) - 1
        elif i and a[i] - x > x - a[i - 1]:
            i = i - 1
        return i  # return (i, a[i])

    @staticmethod
    def get_closest(array, values):
        # make sure array is a numpy array
        array = np.array(array)

        # get insert positions
        idxs = np.searchsorted(array, values, side="left")

        # find indexes where previous index is closer
        prev_idx_is_less = ((idxs == len(array)) | (np.fabs(values - array[np.maximum(idxs - 1, 0)]) < np.fabs(
            values - array[np.minimum(idxs, len(array) - 1)])))
        idxs[prev_idx_is_less] -= 1

        return array[idxs]
