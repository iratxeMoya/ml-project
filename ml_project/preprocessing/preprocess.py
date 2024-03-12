import pretty_errors
from datetime import datetime

pretty_errors.config.display_timestamp = True
pretty_errors.config.timestamp_function = lambda: datetime.now().strftime('%Y-%m-%d %H:%M:%S')

from dataset import Dataset
from dvc.api import params_show
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle as pkl
from icecream import ic as print
print.configureOutput(prefix=f'{datetime.now()}|> ')

if __name__ == '__main__':
    params = params_show()["split"]
    raw_data_path = 'data/bbc-text.csv'

    df = pd.read_csv(raw_data_path)
    train_size = params['train']
    test_size = params['test'] / (1 - train_size)

    # Primero, divide en entrenamiento y un conjunto temporal (resto)
    df_train, temp_df = train_test_split(df, train_size=train_size, random_state=42)

    # Luego, divide el conjunto temporal en prueba y validación
    df_test, df_val = train_test_split(temp_df, test_size=test_size, random_state=42)
    del temp_df

    train, val, test = Dataset(df_train), Dataset(df_val), Dataset(df_test)
    datasets = {
        'train': train,
        'test': test,
        'validation': val
    }

    with open('data/datasets.pkl', 'wb') as f:
        pkl.dump(datasets, f)
        
