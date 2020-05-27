import argparse

"""
Author: S.LEGEAY, intern at LaSEEB
e-mail: legeay.simon.sup@gmail.com


# Log data from LSL inLet
"""

from pylsl import StreamInlet, resolve_stream
import threading


def get_marker_stream():
    results = resolve_stream("name", "MarkerStream")

    # open an inlet so we can read the stream's data (and meta-data)
    inlet = StreamInlet(results[0])

    # get the full stream info (including custom meta-data) and dissect it
    info = inlet.info()
    print("The stream's XML meta-data is: ")
    print(info.as_xml())


def get_openvibeSignal():
    results = resolve_stream("name", "openvibeSignal")
    # open an inlet so we can read the stream's data (and meta-data)
    inlet = StreamInlet(results[0])

    # get the full stream info (including custom meta-data) and dissect it
    info = inlet.info()
    print("The stream's XML meta-data is: ")
    print(info.as_xml())


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="# Log data from LSL inLet")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="verbose mode")
    parser.add_argument(
        "-o",
        "--format",
        choices=['xdf', 'vhdr'],
        help="Specify the output format",
        required=True)

    args = parser.parse_args()
    # creating thread
    t1 = threading.Thread(target=get_marker_stream)
    t2 = threading.Thread(target=get_openvibeSignal)

    # starting thread 1
    t1.start()
    # starting thread 2
    t2.start()

    # wait until thread 1 is completely executed
    t1.join()
    # wait until thread 2 is completely executed
    t2.join()

    # both threads completely executed
    print("Done!")
