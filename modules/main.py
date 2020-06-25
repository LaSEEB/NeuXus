import sys

import argparse
import logging
from datetime import datetime

sys.path.append('..')

from modules.pipeline import run


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="# LaSEEB BCI main script")
    parser.add_argument(
        "pipeline",
        help="Path to the pipeline file")
    parser.add_argument(
        "-f",
        "--file",
        action="store_true",
        help="Store logs in a log file, default is on cmd window")
    parser.add_argument(
        "-l",
        "--loglevel",
        choices=['DEBUG', 'INFO'],
        help="Specify the output format",
        default='INFO')
    args = parser.parse_args()

    numeric_level = getattr(logging, args.loglevel.upper(), None)

    if args.file:
        file_path = str(datetime.now().strftime("%d-%m-%Y %Hh%M")) + '.log'
        print(f'Store logs in {file_path}')
        logging.basicConfig(
            filename=file_path,
            filemode='w',
            format='[%(levelname)s] %(asctime)s %(message)s',
            level=numeric_level)
    else:
        logging.basicConfig(
            format='[%(levelname)s] %(message)s',
            level=numeric_level)
    # initialize the nodes
    exec(open(args.pipeline).read())
    # run the pipeline
    run()
