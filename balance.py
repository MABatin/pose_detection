import pandas as pd

from imblearn.over_sampling import RandomOverSampler


def balance(dataset):
    # Create X features and y labels
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]

    # Use Random over sampling for balancing
    oversampling = RandomOverSampler()

    X, y = oversampling.fit_resample(X, y)
    # balanced_dataset = X[0].to_frame().join(y.to_frame())
    # balanced_dataset = balanced_dataset.join(X.iloc[:, 1:])
    balanced_dataset = X.join(y.to_frame())

    return balanced_dataset


def main():
    in_csv = 'results/unbalanced_keypoints.csv'
    out_csv = 'results/balanced_keypoints.csv'

    dataset = pd.read_csv(in_csv, header=None, delimiter=',')

    balanced_dataset = balance(dataset)

    balanced_dataset.to_csv(out_csv, index=False, header=None)


if __name__ == '__main__':
    main()
