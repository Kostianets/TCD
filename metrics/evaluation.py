# -----------------------------------------------------------
# Implementácia metrík ako samostatných funkcií
# -----------------------------------------------------------
def accuracy_metric(y_true, y_pred):
    """
    Vypočíta presnosť (accuracy).
    
    Parameters
    ----------
    - y_true: zoznam skutočných hodnôt
    - y_pred: zoznam predikovaných hodnôt
    
    Returns
    -------
    - presnosť modelu
    """
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)

def precision_metric(y_true, y_pred, positive=1, average='weighted'):
    """
    Vypočíta presnosť (precision) s podporou pre priemerovanie ako v sklearn.
    
    Parameters
    ----------
    - y_true: zoznam skutočných hodnôt
    - y_pred: zoznam predikovaných hodnôt
    - positive: hodnota pozitívnej triedy (štandardne 1), použité len pre average='binary'
    - average: typ priemerovania ('binary' alebo 'weighted', štandardne 'weighted')
    
    Returns
    -------
    - precision (buď pre pozitívnu triedu alebo vážený priemer)
    """
    if average == 'binary':
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == positive and pred == positive)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true != positive and pred == positive)
        return tp / (tp + fp) if (tp + fp) > 0 else 0
    elif average == 'weighted':
        classes = sorted(set(y_true))
        precisions = []
        class_counts = []
        for cls in classes:
            tp = sum(1 for true, pred in zip(y_true, y_pred) if true == cls and pred == cls)
            fp = sum(1 for true, pred in zip(y_true, y_pred) if true != cls and pred == cls)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            precisions.append(precision)
            class_counts.append(sum(1 for true in y_true if true == cls))
        total = sum(class_counts)
        return sum(p * count / total for p, count in zip(precisions, class_counts))
    else:
        raise ValueError("Supported averaging types are 'binary' or 'weighted'")

def recall_metric(y_true, y_pred, positive=1, average='weighted'):
    """
    Vypočíta návratnosť (recall) s podporou pre priemerovanie ako v sklearn.
    
    Parameters
    ----------
    - y_true: zoznam skutočných hodnôt
    - y_pred: zoznam predikovaných hodnôt
    - positive: hodnota pozitívnej triedy (štandardne 1), použité len pre average='binary'
    - average: typ priemerovania ('binary' alebo 'weighted', štandardne 'weighted')
    
    Returns
    -------
    - recall (buď pre pozitívnu triedu alebo vážený priemer)
    """
    if average == 'binary':
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == positive and pred == positive)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == positive and pred != positive)
        return tp / (tp + fn) if (tp + fn) > 0 else 0
    elif average == 'weighted':
        classes = sorted(set(y_true))
        recalls = []
        class_counts = []
        for cls in classes:
            tp = sum(1 for true, pred in zip(y_true, y_pred) if true == cls and pred == cls)
            fn = sum(1 for true, pred in zip(y_true, y_pred) if true == cls and pred != cls)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            recalls.append(recall)
            class_counts.append(sum(1 for true in y_true if true == cls))
        total = sum(class_counts)
        return sum(r * count / total for r, count in zip(recalls, class_counts))
    else:
        raise ValueError("Supported averaging types are 'binary' or 'weighted'")

def f1_metric(y_true, y_pred, positive=1, average='weighted'):
    """
    Vypočíta F1 mieru s podporou pre priemerovanie ako v sklearn.
    
    Parameters
    ----------
    - y_true: zoznam skutočných hodnôt
    - y_pred: zoznam predikovaných hodnôt
    - positive: hodnota pozitívnej triedy (štandardne 1), použité len pre average='binary'
    - average: typ priemerovania ('binary' alebo 'weighted', štandardne 'weighted')
    
    Returns
    -------
    - F1 miera (buď pre pozitívnu triedu alebo vážený priemer)
    """
    if average == 'binary':
        prec = precision_metric(y_true, y_pred, positive=positive, average='binary')
        rec = recall_metric(y_true, y_pred, positive=positive, average='binary')
        return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    elif average == 'weighted':
        classes = sorted(set(y_true))
        f1_scores = []
        class_counts = []
        for cls in classes:
            prec = precision_metric(y_true, y_pred, positive=cls, average='binary')
            rec = recall_metric(y_true, y_pred, positive=cls, average='binary')
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            f1_scores.append(f1)
            class_counts.append(sum(1 for true in y_true if true == cls))
        total = sum(class_counts)
        return sum(f1 * count / total for f1, count in zip(f1_scores, class_counts))
    else:
        raise ValueError("Supported averaging types are 'binary' or 'weighted'")