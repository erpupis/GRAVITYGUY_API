from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential

def create_model(X_train):

    # Initialize the model
    model = Sequential()

    # Input layer
    model.add(Dense(32, activation='relu', input_shape=(X_train.shape[1],)))

    # Hidden layers
    model.add(Dense(16, activation='relu'))

    # Output layer - Using sigmoid for binary classification
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model
