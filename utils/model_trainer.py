import random
from algorithm.bagging import BaggingClassifier
from algorithm.naive_bayes import SimpleNaiveBayesClassifier
from utils.data_loader import load_data
from metrics.evaluation import accuracy_metric, precision_metric, recall_metric, f1_metric
from utils.model_saver import auto_save_best_model, load_best_model

def train_model(texts, labels):
    """
    Trénuje model na základe zoznamu textov a príslušných labelov.
    Používa 70% dát na trénovanie, 15% na testovanie a 15% na evaluáciu.
    Uloží evaluačný graf.
    
    Returns:
    - model: natrénovaný model
    - metrics: slovník metrík (testovacia množina)
    """
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts_shuffled, labels_shuffled = zip(*combined)
    n_samples = len(texts_shuffled)
    
    train_end = int(0.70 * n_samples)
    test_end = int(0.85 * n_samples)
    
    X_train = list(texts_shuffled[:train_end])
    y_train = list(labels_shuffled[:train_end])
    
    X_test = list(texts_shuffled[train_end:test_end])
    y_test = list(labels_shuffled[train_end:test_end])
    
    X_eval = list(texts_shuffled[test_end:])
    y_eval = list(labels_shuffled[test_end:])
    
    model = BaggingClassifier(base_estimator=SimpleNaiveBayesClassifier,
                              n_estimators=10,
                              max_samples=len(X_train))
    model.fit(X_train, y_train)
    
    predictions_test = model.predict(X_test)
    test_metrics = {
        "Accuracy": accuracy_metric(y_test, predictions_test),
        "Precision": precision_metric(y_test, predictions_test),
        "Recall": recall_metric(y_test, predictions_test),
        "F1 Score": f1_metric(y_test, predictions_test)
    }
    
    predictions_eval = model.predict(X_eval)
    eval_metrics = {
        "Accuracy": accuracy_metric(y_eval, predictions_eval),
        "Precision": precision_metric(y_eval, predictions_eval),
        "Recall": recall_metric(y_eval, predictions_eval),
        "F1 Score": f1_metric(y_eval, predictions_eval)
    }
    
    return model, test_metrics, eval_metrics

def evaluate_model(model, texts, labels):
    """
    Vyhodnotí existujúci model pomocou 70%/15%/15% splitu (trénovacia časť nie je použitá)
    a uloží evaluačný graf.
    
    Returns:
    - metrics: slovník metrík na testovacej množine
    """
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts_shuffled, labels_shuffled = zip(*combined)
    n_samples = len(texts_shuffled)
    
    train_end = int(0.70 * n_samples)
    test_end = int(0.85 * n_samples)
    
    X_test = list(texts_shuffled[train_end:test_end])
    y_test = list(labels_shuffled[train_end:test_end])
    
    X_eval = list(texts_shuffled[test_end:])
    y_eval = list(labels_shuffled[test_end:])
    
    predictions_test = model.predict(X_test)
    test_metrics = {
        "Accuracy": accuracy_metric(y_test, predictions_test),
        "Precision": precision_metric(y_test, predictions_test),
        "Recall": recall_metric(y_test, predictions_test),
        "F1 Score": f1_metric(y_test, predictions_test)
    }
    
    predictions_eval = model.predict(X_eval)
    eval_metrics = {
        "Accuracy": accuracy_metric(y_eval, predictions_eval),
        "Precision": precision_metric(y_eval, predictions_eval),
        "Recall": recall_metric(y_eval, predictions_eval),
        "F1 Score": f1_metric(y_eval, predictions_eval)
    }
    
    return eval_metrics

def get_trained_model(label: str):
    """
    Načíta dáta a pokúsi sa načítať najlepší uložený model pre daný label.
    Ak model existuje, vyhodnotí sa na aktuálnom splite.
    Potom sa natrénuje nový model a porovná F1 skóre s existujúcim najlepším.
    Ak je nový model lepší, uloží sa ako najlepší a jeho výsledky
    sa uložia do evaluačného grafu s príponou "_best".
    
    Returns:
    - model: najlepší model pre daný label
    - metrics: slovník metrík na testovacej množine
    """
    texts, labels = load_data(label)
    best_model, best_metrics = load_best_model(f"models/best_model_{label}.pkl")

    if best_model is not None:
        test_metrics = evaluate_model(best_model, texts, labels)
        return best_model, test_metrics
    else:
        model, test_metrics = train_model(texts, labels, label)
        auto_save_best_model(model, test_metrics, f"models/best_model_{label}.pkl")
        return model, test_metrics