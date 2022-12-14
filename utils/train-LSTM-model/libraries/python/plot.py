import numpy as np
from matplotlib import pyplot as plt
from collections import deque
from bisect import bisect_left

class Plot:
    def __init__(self, xdur, xmargin, slide, lines):

        self.xdur = xdur
        self.xmargin = xmargin

        plt.ion()

        legend_locs = []

        # Create + set axes
        self.axis = []
        subplots_unique = Plot.unique([line['subplot'] for line in lines])
        self.fh, axis = plt.subplots(len(subplots_unique), 1)
        if not isinstance(axis, np.ndarray):
            axis = [axis]
        for i, axi in enumerate(axis):
            axi.title.set_text(subplots_unique[i])
            axi.set_ylabel('Amplitude [uV]')
            axi.set_xlim(0, self.xdur + self.xmargin)
            if i < len(axis) - 1:
                pass
                # axi.set_xticklabels([])
            else:
                axi.set_xlabel('Time [s]')
            self.axis.append(axi)
            legend_locs.append(1)


        # Set lines
        ylims_unique = Plot.unique([line['ylim'] for line in lines if line['subplot'] == subplots_unique[0]])
        si = 0
        siyi = 0
        axis_id = si
        for i, _ in enumerate(lines):

            if lines[i]['subplot'] is not subplots_unique[si]:
                si += 1
                ylims_unique = Plot.unique([line['ylim'] for line in lines if line['subplot'] == subplots_unique[si]])
                siyi = 0
                axis_id = si

            if lines[i]['ylim'] != ylims_unique[siyi]:
                axi = axis[si + siyi].twinx()
                self.axis.append(axi)
                legend_locs.append(2)
                axis_id = len(self.axis) - 1
                siyi += 1

            marker = None
            ls = '-'
            lw = 0.5
            fillstyle = 'full'
            if 'marker' in lines[i]:
                marker = lines[i]['marker']

            if 'ls' in lines[i]:
                ls = lines[i]['ls']

            if 'lw' in lines[i]:
                lw = lines[i]['lw']

            if 'fillstyle' in lines[i]:
                fillstyle = lines[i]['fillstyle']

            # lines[i]['axis'] = axi
            lines[i]['axis_id'] = axis_id
            lines[i]['signal'] = deque(maxlen=int(xdur * lines[i]['fs']))
            lines[i]['times'] = deque(maxlen=int(xdur * lines[i]['fs']))
            lines[i]['line'],  = self.axis[lines[i]['axis_id']].plot([], [], lw=lw, color=lines[i]['col'], label=lines[i]['label'], ls=ls, marker=marker, fillstyle=fillstyle)
            self.axis[axis_id].set_ylim(lines[i]['ylim'])

        self.lines = lines
        self.names = [line['name'] for line in lines]
        self.slide_all_based_on_time_from = self.names.index(slide)
        # for i, axi in enumerate(self.axis):
        #     axi.legend(loc=legend_locs[i])

        for subplot in subplots_unique:
            subplot_lines = [line for line in lines if line['subplot'] == subplot]
            # labs = [self.axis[line['axis_id']].get_label() for line in lines if line['subplot'] == subplots_unique[si]]
            self.axis[subplot_lines[0]['axis_id']].legend([line['line'] for line in subplot_lines], [line['line'].get_label() for line in subplot_lines], loc=0)

            # ax.legend([ for subplot_line in subplot_lines], labs, loc=0)



    @staticmethod
    def unique(seq):
        # order preserving
        checked = []
        for e in seq:
            if e not in checked:
                checked.append(e)
        return checked

    def update(self, name, chunks, chan):
        i = self.names.index(name)
        for chunk in chunks:
            self.lines[i]['times'].extend(chunk.index.to_numpy())
            self.lines[i]['signal'].extend(chunk[chan].to_numpy())
        self.lines[i]['line'].set_data(self.lines[i]['times'], self.lines[i]['signal'])

    def update_marker_points(self,name_of_marked_line,name,chunks,marker):
        i = self.names.index(name)
        j = self.names.index(name_of_marked_line)
        times = []
        amps = []
        for chunk in chunks:
            if chunk.iloc[0].values in marker:
                time = chunk.index.to_numpy()
                times.append(time)
                amps.append(self.lines[j]['signal'][bisect_left(np.asarray(self.lines[j]['times']), time)])
        if len(times):
            self.lines[i]['line'].set_data(times, amps)

    def update_marker_lines(self,name,chunks,marker):
        i = self.names.index(name)
        times = []
        amps = []
        start_time = 0
        for chunk in chunks:
            if chunk.iloc[0].values in marker:
                time = chunk.index.to_numpy()
                times.extend([start_time, time, time])
                amps.extend([np.nan, *self.lines[i]['ylim']])
                start_time = time
        if len(times):
            times[0] = times[1] - 1
            self.lines[i]['line'].set_data(times, amps)

    def slide(self):
        last_time = self.lines[self.slide_all_based_on_time_from]['times'][-1]
        if last_time >= self.xdur:
            for i, axi in enumerate(self.axis):
                self.axis[i].set_xlim(last_time - self.xdur, last_time + self.xmargin)

    def draw(self):
        self.fh.canvas.draw()
        self.fh.canvas.flush_events()

