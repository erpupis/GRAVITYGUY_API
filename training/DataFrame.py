from sklearn.preprocessing import MinMaxScaler
from daos.input_dao import InputDao
from core.database import Database
from imblearn.over_sampling import SMOTE
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN
from joblib import Parallel, delayed


def initialize_database():
    db = Database()
    db.initialize()
    return db


def fetch_player_data(db, player_name):
    dao = InputDao(db)
    return dao.get_inputs(player_name)

def multiply_columns_by_100(df, columns_to_multiply):
    df[columns_to_multiply] = df[columns_to_multiply] * 100

def handle_infinite_values(df):
    df.replace([np.inf], 1000, inplace=True)
    df.replace([-np.inf], -1000, inplace=True)


def scale_features(df):
    scaler = MinMaxScaler()
    columns_to_scale = [
        'raycast_0', 'raycast_30', 'raycast_45', 'raycast_315',
        'raycast_330', 'collect_angle', 'collect_length', 'gravity_dir', 'speed'
    ]
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])


def split_features_labels(df):
    columns_to_drop = ['switch_gravity']
    X = df.drop(columns_to_drop, axis=1)
    y = df['switch_gravity']
    return X, y


def perform_train_test_split(X, y):
    return train_test_split(X, y, test_size=0.2)


def cast_data_types(X_train, X_test, y_train, y_test):
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')

    return X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()



def process_data(player_name):
    db = initialize_database()
    df = fetch_player_data(db, player_name)

    if df is None:
        print("DataFrame is None. Cannot proceed.")
        return

    handle_infinite_values(df)
    multiply_columns_by_100(df, ['raycast_0_horizontal', 'raycast_30_horizontal','raycast_45_horizontal','raycast_315_horizontal','raycast_330_horizontal'])
    scale_features(df)

    X, y = split_features_labels(df)
    X_train, X_test, y_train, y_test = perform_train_test_split(X, y)

    X_train, X_test, y_train, y_test = cast_data_types(X_train, X_test, y_train, y_test)
    return X_train, X_test, y_train, y_test




