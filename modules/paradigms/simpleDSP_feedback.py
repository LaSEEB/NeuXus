import sys

from time import time
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append('.')
sys.path.append('../..')

from modules.port.port import Port
from modules.node.node import (Send, Receive, ButterFilter, Epoching, Averaging, ApplyFunction, ChannelSelector)


if __name__ == '__main__':

    # for observation via plt
    observe_plt = False

    nominal_srate = 250

    # initialize the pipeline
    port0 = Port()
    lsl_reception = Receive(port0, 'name', 'openvibeSignal')  # or (port0, 'type', 'signal')
    port1 = Port()
    select = ChannelSelector(port0, port1, 'index', [24])
    port2 = Port()
    butter_filter = ButterFilter(port1, port2, 8, 12)
    port3 = Port()
    apply_function = ApplyFunction(port2, port3, lambda x: x**2)
    port4 = Port()
    port4.set_channels(port1.channels)
    port5 = Port()
    port5.set_channels(port1.channels)

    epoch = Epoching(port3, port4, 1)
    average = Averaging(port4, port5)
    lsl_send = Send(port5, 'mySignalEpoched', 1)
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
        port4.clear()
        port5.clear()

        lsl_reception.update()
        select.update()
        butter_filter.update()
        apply_function.update()
        lsl_send2.update()
        epoch.update()
        average.update()
        lsl_send.update()

        calc_endtime = time()
        calc_time = calc_endtime - calc_starttime

        # print(f'{ int(calc_time* 1000)}ms for {port1.length} treated rows ({port1.length * len(port1.channels)} data)')

        # for dev
        """it += 1
                                data1 = pd.concat([data1, port1.data])
                                data = pd.concat([data, port2.data])
                                if observe_plt and it == 150:
                                    plt.plot(data.iloc[:, 0:1].values)
                                    plt.plot(data1.iloc[:, 0:1].values)
                                    plt.show()"""

        #try:
        #    sleep(t - calc_time)
        #except Exception as err:
        #    print(err)
    # TO DO terminate
