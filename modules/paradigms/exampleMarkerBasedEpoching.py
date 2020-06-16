import sys

from time import time
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append('.')
sys.path.append('../..')

from modules.port.port import Port
from modules.node.node import (Send, Receive, ButterFilter, TimeBasedEpoching, Averaging, ApplyFunction, ChannelSelector, MarkerBasedEpoching)


if __name__ == '__main__':

    # for observation via plt
    observe_plt = False

    # initialize the pipeline

    m_port0 = Port()
    lsl_marker_reception = Receive(m_port0, 'type', 'Markers')
    port0 = Port()
    lsl_reception = Receive(port0, 'name', 'openvibeSignal')
    port2 = Port()
    epoch = MarkerBasedEpoching(port0, port2, m_port0)
    lsl_send = Send(port2, 'mySignalEpoched')

    '''# for dev
    data = pd.DataFrame([])
    data1 = pd.DataFrame([])
    # count iteration
    it = 0'''

    # run the pipeline
    try:
        while True:
            calc_starttime = time()

            # clear port
            port0.clear()
            m_port0.clear()
            port2.clear()

            lsl_marker_reception.update()
            lsl_reception.update()
            epoch.update()
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
    except KeyboardInterrupt:
        print('Stop processing')
        exit()
    # TO DO terminate
