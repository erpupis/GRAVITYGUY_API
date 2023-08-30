from training.networkModel import create_model
from training.DataFrame import process_data
import datetime
import base64
def train(player_name: str):
    train_start = datetime.datetime.now().isoformat()

    X_train, X_test, y_train, y_test = process_data(player_name)
    model = create_model(X_train)
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")

    model.save_weights("weights.h5")

    # Convert the weights to a Base64-encoded string
    with open("weights.h5", "rb") as f:
        weights_data = f.read()

    weights_base64 = base64.b64encode(weights_data).decode('utf-8')
    train_end = datetime.datetime.now().isoformat()

    return train_start, train_end, weights_base64, test_accuracy



