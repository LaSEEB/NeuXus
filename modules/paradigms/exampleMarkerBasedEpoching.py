import sys

from time import time
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append('.')
sys.path.append('../..')

from modules.port.port import Port
import modules.node.node as node

if __name__ == '__main__':

    # for observation via plt
    observe_plt = False

    # initialize the pipeline

    m_port0 = Port()
    lsl_marker_reception = node.LslReceive(m_port0, 'type', 'Markers')
    port0 = Port()
    lsl_reception = node.LslReceive(port0, 'name', 'openvibeSignal')
    port1 = Port()
    select = node.ChannelSelector(port0, port1, 'index', [24])
    port2 = Port()
    epoch_right = node.StimulationBasedEpoching(port1, port2, m_port0, 770, 0.125, 1)
    port3 = Port()
    epoch_left = node.StimulationBasedEpoching(port1, port3, m_port0, 769, 0.125, 1)
    lsl_send = node.LslSend(port2, 'mySignalEpoched')

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
            port1.clear()
            port2.clear()
            port3.clear()

            lsl_marker_reception.update()
            lsl_reception.update()
            select.update()
            epoch_left.update()
            epoch_right.update()
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
            if len(port2._data) > 0:
                plt.plot(port2._data[0].iloc[:, 0:1].values)
                plt.show()
            if len(port3._data) > 0:
                plt.plot(port3._data[0].iloc[:, 0:1].values)
                plt.show()

            #try:
            #    sleep(t - calc_time)
            #except Exception as err:
            #    print(err)
    except KeyboardInterrupt:
        print('Stop processing')
        exit()
    # TO DO terminate
