import sys

from time import time
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append('.')
sys.path.append('../..')

from modules.core.port import Port
import modules.core.node as node

def run():

    # for observation via plt
    observe_plt = False
    '''# for dev
    data = pd.DataFrame([])
    data1 = pd.DataFrame([])
    # count iteration
    it = 0'''

    # run the pipeline
    try:
        while True:
            calc_starttime = time()

            for port in Port.get_instances():
                port.clear()
            for nods in node.Node.get_instances():
                nods.update()

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
