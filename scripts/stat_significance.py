import argparse
import pandas as pd
from scipy import stats
import numpy as np


def cohens_d(a, b):
    diff = np.mean(a) - np.mean(b)
    pooled = np.sqrt((np.std(a)**2 + np.std(b)**2) / 2)
    return diff / pooled


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--file")
    parser.add_argument("--col_a")
    parser.add_argument("--col_b")

    args = parser.parse_args()

    df = pd.read_csv(args.file)

    a = df[args.col_a]
    b = df[args.col_b]

    t, p = stats.ttest_rel(a, b)

    d = cohens_d(a, b)

    print("\nStatistical test result")
    print("-----------------------")
    print("p-value:", p)
    print("Cohen's d:", d)


if __name__ == "__main__":
    main()