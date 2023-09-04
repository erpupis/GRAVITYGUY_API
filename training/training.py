from training.DataFrame import process_data
from training.networkModel import create_model
import datetime
import base64
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve
import math
import matplotlib.pyplot as plt


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
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
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

def evaluate_model(model, X_test, y_test, model_type='nn'):
    if model_type == 'nn':
        _, accuracy = model.evaluate(X_test, y_test)
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


def train(player_name):
    train_start = datetime.datetime.now().isoformat()

    X_train, X_test, y_train, y_test = load_data(player_name)

    # Neural Network
    nn_model = train_nn_model(X_train, y_train)
    nn_accuracy = evaluate_model(nn_model, X_test, y_test, model_type='nn')
    nn_weights_base64 = save_nn_weights(nn_model)
    nn_predictions_raw = nn_model.predict(X_test)

    # Determine the optimal threshold

    precision, recall, thresholds = precision_recall_curve(y_test, nn_predictions_raw)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    best_threshold = thresholds[f1_scores.argmax()]

    nn_predictions = [1 if prob >= best_threshold else 0 for prob in nn_predictions_raw]
    nn_precision = precision_score(y_test, nn_predictions)
    nn_recall = recall_score(y_test, nn_predictions)

    # Random Forest
    rf_model = train_rf_model(X_train, y_train)
    rf_accuracy = evaluate_model(rf_model, X_test, y_test, model_type='rf')
    log_feature_importance(rf_model)

    # k-Nearest Neighbors
    N = closest_odd_sqrt(X_train.shape[0])
    knn_model = train_knn_model(X_train, y_train, N)
    knn_accuracy = evaluate_model(knn_model, X_test, y_test, model_type='knn')  # 'knn' is just a label for logging

    # Logistic Regression
    log_model = train_log_model(X_train, y_train)
    log_accuracy = evaluate_model(log_model, X_test, y_test, model_type='log')

    train_end = datetime.datetime.now().isoformat()

    # Plot the precision-recall curve
    plt.figure(figsize=(10, 7))
    plt.plot(recall, precision, marker='.', label='Neural Network')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Neural Network Test Accuracy: {nn_accuracy}")
    print(f"Neural Network Precision: {nn_precision}")
    print(f"Neural Network Recall: {nn_recall}")
    print(f"nn raw predictions: {nn_predictions_raw}")
    print(f"nn predictions: {nn_predictions}")
    print(f"Optimal threshold for Neural Network: {best_threshold}")
    print(f"Random Forest Test Accuracy: {rf_accuracy}")
    print(f"k-NN Test Accuracy: {knn_accuracy}")
    print(f"Logistic Regression Test Accuracy: {log_accuracy}")

    return train_start, train_end, nn_weights_base64, nn_accuracy, nn_precision

#for testing
#if __name__ == "__main__":
#    train('excale')
