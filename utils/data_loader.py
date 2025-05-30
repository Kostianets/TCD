import pandas as pd

def load_data(label: str):
    """
    Načíta dáta zo súboru a vráti zoznam textov a príslušných labelov.

    Parameters
    ----------
    - label: názov stĺpca, ktorý obsahuje labely

    Returns
    -------
    - texts: zoznam textov
    - labels: zoznam labelov
    """
    data = pd.read_csv("data/youtoxic_english_1000.csv")
    texts = data['Text'].tolist()
    labels = data[label].tolist()
    return texts, labels