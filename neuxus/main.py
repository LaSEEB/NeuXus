import os

import argparse
import logging
from datetime import datetime

from neuxus.pipeline import run


def main():
    parser = argparse.ArgumentParser(
        description="# Sygnal main script")
    parser.add_argument(
        "pipeline",
        help="Path to the pipeline script file")
    parser.add_argument(
        "-f",
        "--file",
        action="store_true",
        help="Store logs in a log file, default is on cmd window")
    parser.add_argument(
        "-l",
        "--loglevel",
        choices=['DEBUG', 'INFO'],
        help="Specify the log level, default is INFO",
        default='INFO')
    parser.add_argument(
        "-e",
        "--example",
        action="store_true",
        help="To run an example from sygnal")
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
    if args.example:
        file = os.path.split(os.path.split(__file__)[0])[0] + '/examples/' + args.pipeline
    else:
        file = args.pipeline

    # run the pipeline
    logging.info(f'Run {file}')
    run(file)
