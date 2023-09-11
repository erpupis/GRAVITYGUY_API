from training.DataFrame import process_data
from training.networkModel import create_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.utils import compute_class_weight
from sklearn.utils.class_weight import compute_sample_weight
import onnxruntime as rt
import numpy as np
import matplotlib.pyplot as plt
import math
import datetime
import base64
import tf2onnx
import io


def load_data(player_name):
    return process_data(player_name)

def closest_odd_sqrt(N):
    sqrt_N = math.sqrt(N)
    rounded_sqrt_N = round(sqrt_N)

    if rounded_sqrt_N % 2 == 0:
        if rounded_sqrt_N - 1 >= sqrt_N - 0.5:
            rounded_sqrt_N -= 1  # Subtract 1 to make it odd
        else:
            rounded_sqrt_N += 1  # Add 1 to make it odd

    return rounded_sqrt_N

def train_nn_model(X_train, y_train):
    model = create_model(X_train)
    class_weights = {0: 1, 1: 1}
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, class_weight=class_weights)
    return model


def train_rf_model(X_train, y_train):
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    return clf

def train_knn_model(X_train, y_train, n_neighbors):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn

def train_log_model(X_train, y_train):
    log = LogisticRegression()
    log.fit(X_train, y_train)
    return log

def evaluate_model(model, X_test, y_test, threshold, model_type='nn'):
    if model_type == 'nn':
        nn_predictions_raw = model.predict(X_test)
        nn_predictions = [1 if prob >= threshold else 0 for prob in nn_predictions_raw]
        accuracy = np.mean(nn_predictions == y_test)
    else:  # Assuming it's a sklearn model if it's not a neural network
        accuracy = model.score(X_test, y_test)
    return accuracy

def save_nn_weights(model):
    model.save_weights("nn_weights.h5")
    with open("nn_weights.h5", "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def log_feature_importance(model):
    feature_importances = model.feature_importances_
    for i, importance in enumerate(feature_importances):
        print(f"Feature {i}: {importance}")


def keras_to_onnx_bytes(keras_model):
    # Convert the Keras model to ONNX format
    model_proto, _ = tf2onnx.convert.from_keras(
        keras_model,
        opset=13
    )

    # Save the ONNX model to a bytes buffer
    buffer = io.BytesIO()
    buffer.write(model_proto.SerializeToString())

    # Return the bytes buffer to the caller
    return buffer.getvalue()


def test_onnx_model(model_bytes, X_test, y_test, threshold):
    # Load the ONNX model
    sess = rt.InferenceSession(io.BytesIO(model_bytes).read())
    input_name = sess.get_inputs()[0].name

    # Convert X_test to float32
    X_test_float32 = X_test.astype(np.float32)

    # Predict
    predicted = sess.run(None, {input_name: X_test_float32})[0]

    # Assuming the output of the model is in probabilities and you need to round off to get the class
    predicted_labels = (predicted >= threshold).astype(int)

    # Flatten predicted_labels for evaluation
    predicted_labels_flat = predicted_labels.flatten()

    # Calculate precision, recall, and accuracy
    true_positives = np.sum((predicted_labels_flat == 1) & (y_test == 1))
    false_positives = np.sum((predicted_labels_flat == 1) & (y_test == 0))
    true_negatives = np.sum((predicted_labels_flat == 0) & (y_test == 0))
    false_negatives = np.sum((predicted_labels_flat == 0) & (y_test == 1))

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)

    print("onnx labels: ", predicted_labels_flat[:30])
    print(f"Accuracy of onnx: {accuracy * 100:.2f}%")
    print(f"Precision of onnx: {precision * 100:.2f}%")
    print(f"Recall of onnx: {recall * 100:.2f}%")
    print("true positives: ", true_positives)

    return predicted_labels_flat, accuracy, precision, predicted


def train(player_name):
    train_start = datetime.datetime.now().isoformat()

    X_train, X_test, y_train, y_test = load_data(player_name)

    # Neural Network
    nn_model = train_nn_model(X_train, y_train)
    nn_predictions_raw = nn_model.predict(X_test)
    onnx_model = keras_to_onnx_bytes(nn_model)

    # Determine the optimal threshold

    precision, recall, thresholds = precision_recall_curve(y_test, nn_predictions_raw)
    denominator = precision + recall
    f1_scores = 2 * (precision * recall) / np.where(denominator == 0, 1, denominator)
    f1_scores[denominator == 0] = 0.0
    best_threshold = thresholds[f1_scores.argmax()]
    nn_accuracy = evaluate_model(nn_model, X_test, y_test, best_threshold, model_type='nn')

    nn_predictions = [1 if prob >= best_threshold else 0 for prob in nn_predictions_raw]
    nn_precision = precision_score(y_test, nn_predictions)
    nn_recall = recall_score(y_test, nn_predictions)

    # Random Forest
    rf_model = train_rf_model(X_train, y_train)
    rf_accuracy = evaluate_model(rf_model, X_test, y_test, best_threshold, model_type='rf')
    log_feature_importance(rf_model)

    train_end = datetime.datetime.now().isoformat()

    # Plot the precision-recall curve
    plt.figure(figsize=(10, 7))
    plt.plot(recall, precision, marker='.', label='Neural Network')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    #plt.show()

    # test onnx
    onnx_predicted_labels, onnx_accuracy, onnx_precision, predicted = test_onnx_model(onnx_model, X_test, y_test, best_threshold)
    equal_count = np.sum(onnx_predicted_labels == nn_predictions)

    print(f"Neural Network Test Accuracy: {nn_accuracy}")
    print(f"Neural Network Precision: {nn_precision}")
    print(f"Neural Network Recall: {nn_recall}")
    print(f"Optimal threshold for Neural Network: {best_threshold}")
    print("nn labels: " , nn_predictions[:30])
    print("EQUAL VALUES ARE:", equal_count, "nn length", len(nn_predictions), "onnx length", len(onnx_predicted_labels))

    return train_start, train_end, onnx_model, nn_accuracy, nn_precision

#for testing
#if __name__ == "__main__":
#    train('excale')
