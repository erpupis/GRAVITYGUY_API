from sklearn.preprocessing import MinMaxScaler
from daos.input_dao import InputDao
from core.database import Database
import numpy as np
from sklearn.model_selection import train_test_split


def initialize_database():
    db = Database()
    db.initialize()
    return db


def fetch_player_data(db, player_name):
    dao = InputDao(db)
    return dao.get_inputs(player_name)


def handle_infinite_values(df):
    df.replace([np.inf], 300, inplace=True)
    df.replace([-np.inf], -300, inplace=True)


def scale_features(df):
    scaler = MinMaxScaler()
    columns_to_scale = [
        'raycast_0', 'raycast_30', 'raycast_45', 'raycast_315',
        'raycast_330', 'collect_angle', 'collect_length', 'gravity_dir'
    ]
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])


def split_features_labels(df):
    X = df.drop('switch_gravity', axis=1)
    y = df['switch_gravity']
    return X, y


def perform_train_test_split(X, y):
    return train_test_split(X, y, test_size=0.2)


def cast_data_types(X_train, X_test, y_train, y_test):
    for X in [X_train, X_test]:
        X['on_ground_top'] = X['on_ground_top'].astype('int32')
        X['on_ground_bot'] = X['on_ground_bot'].astype('int32')
        X.astype('float32')

    y_train = y_train.astype('int32')
    y_test = y_test.astype('int32')

    return X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()


def process_data(player_name):
    db = initialize_database()
    df = fetch_player_data(db, player_name)

    handle_infinite_values(df)
    scale_features(df)

    X, y = split_features_labels(df)
    X_train, X_test, y_train, y_test = perform_train_test_split(X, y)

    X_train, X_test, y_train, y_test = cast_data_types(X_train, X_test, y_train, y_test)
    print(X_test)
    return X_train, X_test, y_train, y_test

process_data('excale')


