import random
from models.bagging import BaggingClassifier
from models.naive_bayes import SimpleNaiveBayesClassifier
from utils.data_loader import load_data
from metrics.evaluation import accuracy_metric, precision_metric, recall_metric, f1_metric

def train_model(texts, labels):
    # Rozdelíme dáta na 80% trénovacie a 20% testovacie (náhodne)
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts_shuffled, labels_shuffled = zip(*combined)
    split_index = int(0.8 * len(texts_shuffled))
    X_train = list(texts_shuffled[:split_index])
    y_train = list(labels_shuffled[:split_index])
    X_test = list(texts_shuffled[split_index:])
    y_test = list(labels_shuffled[split_index:])

    # Inicializácia a trénovanie modelu
    model = BaggingClassifier(base_estimator=SimpleNaiveBayesClassifier,
                              n_estimators=10,
                              max_samples=len(X_train))
    model.fit(X_train, y_train)
    
    # Vyhodnotenie modelu
    predictions = model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_metric(y_test, predictions),
        "Precision": precision_metric(y_test, predictions),
        "Recall": recall_metric(y_test, predictions),
        "F1 Score": f1_metric(y_test, predictions)
    }
    return model, metrics

def get_trained_model():
    texts, labels = load_data()
    model, metrics = train_model(texts, labels)
    return model, metrics