from sklearn.preprocessing import MinMaxScaler
from daos.input_dao import InputDao
from core.database import Database
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from imblearn.pipeline import Pipeline
import numpy as np
from sklearn.model_selection import train_test_split

def initialize_database():
    db = Database()
    db.initialize()
    return db


def hybrid_resample(X, y, desired_ratio=0.5):
    """
    Performs hybrid resampling on the given data.

    Parameters:
    - X: Features
    - y: Labels
    - desired_ratio: The desired ratio of minority class after resampling.

    Returns:
    - X_resampled: Resampled features
    - y_resampled: Resampled labels
    """

    # Calculate the current class distribution
    class_counts = Counter(y)
    majority_class = max(class_counts, key=class_counts.get)
    minority_class = min(class_counts, key=class_counts.get)

    # Oversample using SMOTE to approach the desired ratio
    smote_ratio = (class_counts[majority_class] * desired_ratio) / (1.0 - desired_ratio)
    smote = SMOTE(sampling_strategy={minority_class: int(smote_ratio)})
    X_smote, y_smote = smote.fit_resample(X, y)

    # Undersample the majority class to achieve the desired ratio
    rus_ratio = {majority_class: int(smote_ratio / desired_ratio)}
    rus = RandomUnderSampler(sampling_strategy=rus_ratio)
    X_resampled, y_resampled = rus.fit_resample(X_smote, y_smote)

    return X_resampled, y_resampled

def fetch_player_data(db, player_name):
    dao = InputDao(db)
    return dao.get_inputs(player_name)


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
    scale_features(df)

    X, y = split_features_labels(df)
    X_train, X_test, y_train, y_test = perform_train_test_split(X, y)

    # Add oversampling here
    X_train, y_train = hybrid_sample_data(X_train, y_train)

    X_train, X_test, y_train, y_test = cast_data_types(X_train, X_test, y_train, y_test)
    return X_train, X_test, y_train, y_test




