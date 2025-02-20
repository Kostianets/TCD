import pandas as pd

def load_data():
    data = pd.read_csv("data/youtoxic_english_1000.csv")
    # Predpokladáme, že CSV obsahuje stĺpce "Text" a "IsToxic"
    texts = data['Text'].tolist()
    labels = data['IsToxic'].tolist()
    return texts, labels