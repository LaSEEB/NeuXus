import sys
import os

from time import time
import psutil
import logging

sys.path.append('.')
sys.path.append('../..')

from modules.core.port import Port
import modules.core.node as node

def run():

    pid = os.getpid()
    py = psutil.Process(pid)

    logging.info('Run pipeline')

    # run the pipeline
    try:
        while True:
            # print('CPU ', py.cpu_percent(interval=.0001), '%')
            print('RAM ', int(100 * py.memory_percent()) / 100, '%')
            '''
            p = py.parents()
            for p1 in p:
                print(p1.cpu_percent(interval=.001))
                print(p1.memory_percent())'''
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
