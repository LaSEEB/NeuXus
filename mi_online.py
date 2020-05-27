import argparse

"""
Author: S.LEGEAY, intern at LaSEEB
e-mail: legeay.simon.sup@gmail.com


# TO DO
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="# TO DO")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="verbose mode")
    parser.add_argument(
        "-lo",
        "--lowest_freq",
        type=float,
        help="Specify the lowest working frequency",
        required=True)
    parser.add_argument(
        "-hi",
        "--highest_freq",
        type=float,
        help="Specify the highest working frequency",
        required=True)
    parser.add_argument(
        "-csp",
        "--TO do",
        help="Specify the csp config file",
        required=True)
    parser.add_argument(
        "-c",
        "--classifier",
        help="Specify the classifier config file",
        required=True)
    args = parser.parse_args()

    print(args.lowest_freq, args.highest_freq)

    exit()
