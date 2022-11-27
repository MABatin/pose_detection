import pandas as pd


def main():
    unbalanced_data = pd.read_csv('results/unbalanced_keypoints.csv', header=None, sep=',')
    balanced_data = pd.read_csv('results/balanced_keypoints.csv', header=None, sep=',')

    unbalanced_cnt = unbalanced_data.iloc[:, -1].value_counts()
    print(f'Before balancing:\n'
          f'{unbalanced_cnt}')

    balanced_cnt = balanced_data.iloc[:, -1].value_counts()
    print(f'After balancing:\n'
          f'{balanced_cnt}')


if __name__ == '__main__':
    main()
