import os
import sys

from time import time
import psutil
import logging

sys.path.append('..')

from modules.chunks import IterChunks
from modules.node import Node


def run():

    pid = os.getpid()
    py = psutil.Process(pid)

    logging.info('Run pipeline')

    # run the pipeline
    try:
        while True:
            # print('CPU ', py.cpu_percent(interval=.0001), '%')
            # print('RAM ', int(100 * py.memory_percent()) / 100, '%')
            '''
            p = py.parents()
            for p1 in p:
                print(p1.cpu_percent(interval=.001))
                print(p1.memory_percent())'''
            calc_starttime = time()

            for port in IterChunks.get_instances():
                port.clear()
            for nods in Node.get_instances():
                nods.update()
                nods.update_to_log()

            calc_endtime = time()
            calc_time = calc_endtime - calc_starttime

    except KeyboardInterrupt:
        print('Stop processing')
        exit()
    # TO DO terminate
