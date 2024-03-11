import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def ensure_dataset_scale(df: pd.DataFrame) -> pd.DataFrame:
    df['label'] = df['label'].astype(int)
    # assert that all labels are of the same size
    print(f'Unique labels: {df["label"].unique()}')
    for label in df['label'].unique():
        print(f'Label {label} has {len(df[df["label"] == label])} samples')

    # take x and y columns and scale them to 0-1
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    df['x'] = x_scaler.fit_transform(df['x'].values.reshape(-1, 1))
    df['y'] = y_scaler.fit_transform(df['y'].values.reshape(-1, 1))
    df = df.sample(frac=1).reset_index(drop=True)
    return df


def plot_dataset(df: pd.DataFrame):
    plt.figure(figsize=(5, 5))
    plt.scatter(df['x'], df['y'], c=df['label'], cmap='viridis')
    plt.show()
