import os
import sys

from time import time
import psutil
import logging
import threading
import queue
import numpy as np
from pynput import keyboard

sys.path.append('..')

from modules.chunks import IterChunks
from modules.node import Node


q = queue.Queue()


def on_release(key):
    if key == keyboard.Key.esc:
        # Stop listener
        return False


# Collect events until released
def lis():
    with keyboard.Listener(
            on_release=on_release) as listener:
        listener.join()
        q.put('end')


def run(pipeline):
    """Function to run pipeline"""

    pid = os.getpid()
    py = psutil.Process(pid)

    th = threading.Thread(target=lis)
    th.start()

    exec(open(pipeline).read())

    ports = IterChunks.get_instances()
    logging.debug(f'Name  freq  epoch epochfreq channels')
    for port in ports:
        port.log_parameters()

    logging.info('Run pipeline')

    # run the pipeline
    flag = True
    while flag:
        # logging.debug('New iteration')
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
        try:
            if q.get_nowait() == 'end':
                flag = False
        except queue.Empty:
            pass

    print('Stop processing')
    for nods in Node.get_instances():
        nods.terminate()
    exit()
