import argparse

"""
Author: S.LEGEAY, intern at LaSEEB
e-mail: legeay.simon.sup@gmail.com


# Train the model from data of the file specified
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="# Train the model from data of the file specified")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="verbose mode")
    parser.add_argument(
        "file",
        help="path to the data file")
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
        "-s",
        "--je_sais_pas",
        choices=['csp', '..1', '..2'],
        help="To do",
        required=True)
    parser.add_argument(
        "-c",
        "--classifier",
        choices=['LDA', 'SVM', 'ANN'],
        help="Specify the classifier to used in training",
        required=True)
    args = parser.parse_args()

    print(args.lowest_freq, args.highest_freq)

    exit()
