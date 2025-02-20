# -----------------------------------------------------------
# Implementácia metrík ako samostatných funkcií
# -----------------------------------------------------------
def accuracy_metric(y_true, y_pred):
    """
    Vypočíta presnosť (accuracy).
    
    Parametre:
    - y_true: zoznam skutočných hodnôt
    - y_pred: zoznam predikovaných hodnôt
    
    Návratová hodnota:
    - presnosť modelu
    """
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)

def precision_metric(y_true, y_pred, positive=1):
    """
    Vypočíta presnosť (precision) pre pozitívnu triedu.
    
    Parametre:
    - y_true: zoznam skutočných hodnôt
    - y_pred: zoznam predikovaných hodnôt
    - positive: hodnota, ktorá predstavuje pozitívnu triedu (štandardne 1)
    
    Návratová hodnota:
    - precision pre pozitívnu triedu
    """
    tp = sum(1 for true, pred in zip(y_true, y_pred) if true == positive and pred == positive)
    fp = sum(1 for true, pred in zip(y_true, y_pred) if true != positive and pred == positive)
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def recall_metric(y_true, y_pred, positive=1):
    """
    Vypočíta návratnosť (recall) pre pozitívnu triedu.
    
    Parametre:
    - y_true: zoznam skutočných hodnôt
    - y_pred: zoznam predikovaných hodnôt
    - positive: hodnota, ktorá predstavuje pozitívnu triedu (štandardne 1)
    
    Návratová hodnota:
    - recall pre pozitívnu triedu
    """
    tp = sum(1 for true, pred in zip(y_true, y_pred) if true == positive and pred == positive)
    fn = sum(1 for true, pred in zip(y_true, y_pred) if true == positive and pred != positive)
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def f1_metric(y_true, y_pred, positive=1):
    """
    Vypočíta F1 mieru pre pozitívnu triedu.
    
    Parametre:
    - y_true: zoznam skutočných hodnôt
    - y_pred: zoznam predikovaných hodnôt
    - positive: hodnota, ktorá predstavuje pozitívnu triedu (štandardne 1)
    
    Návratová hodnota:
    - F1 miera
    """
    prec = precision_metric(y_true, y_pred, positive)
    rec = recall_metric(y_true, y_pred, positive)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0