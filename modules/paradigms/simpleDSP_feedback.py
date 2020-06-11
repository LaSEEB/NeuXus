import sys

from time import (time, sleep)
import matplotlib.pyplot as plt
import pandas as pd


sys.path.append('.')
sys.path.append('../..')

from modules.loggers.port import (Port, GroupOfPorts)
from modules.loggers.node import (Send, Receive, ButterFilter, Epoching, Averaging, ApplyFunction, ChannelSelector)


if __name__ == '__main__':

    # for observation via plt
    observe_plt = False

    # initialize the pipeline
    port0 = Port()
    lsl_reception = Receive(port0)
    port1 = Port()
    port1.set_channels(['Channel 25'])
    port2 = Port()
    port2.set_channels(port1.channels)
    port3 = Port()
    port3.set_channels(port1.channels)
    g_port3 = GroupOfPorts()
    port4 = Port()
    port4.set_channels(port1.channels)

    nominal_srate = 250

    select = ChannelSelector(port0, port1, ['Channel 25'])
    butter_filter = ButterFilter(port1, port2, 8, 12, nominal_srate)
    apply_function = ApplyFunction(port2, port3, lambda x: x**2)
    epoch = Epoching(port3, g_port3, 1)
    average = Averaging(g_port3, port4)
    lsl_send = Send(port4, 'mySignalEpoched', 1)
    lsl_send2 = Send(port3, 'mySignalFiltered', nominal_srate)

    # for dev
    data = pd.DataFrame([])
    data1 = pd.DataFrame([])
    # count iteration
    it = 0

    # run the pipeline
    while True:
        calc_starttime = time()

        # clear port
        port0.clear()
        port1.clear()
        port2.clear()
        port3.clear()
        g_port3.clear()
        port4.clear()

        lsl_reception.update()
        if port0.ready():
            select.update()
        if port1.ready():
            butter_filter.update()
        if port2.ready():
            apply_function.update()
        if port3.ready():
            epoch.update()
            lsl_send2.update()
        if g_port3.ready():
            average.update()
        if port4.ready():
            lsl_send.update()

        calc_endtime = time()
        calc_time = calc_endtime - calc_starttime

        print(f'{ int(calc_time* 1000)}ms for {port1.length} treated rows ({port1.length * len(port1.channels)} data)')

        # for dev
        it += 1
        data1 = pd.concat([data1, port1.data])
        data = pd.concat([data, port2.data])
        if observe_plt and it == 150:
            plt.plot(data.iloc[:, 0:1].values)
            plt.plot(data1.iloc[:, 0:1].values)
            plt.show()

        #try:
        #    sleep(t - calc_time)
        #except Exception as err:
        #    print(err)
    # TO DO terminate
