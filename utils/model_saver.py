import pickle
import streamlit as st

def auto_save_best_model(model, metrics, model_filename):
    """
    Automaticky uloží model na disk, ak je lepší ako najlepší uložený model.
    Funkcia bude zmazana neskôr.
    """
    try:
        with open(model_filename, "rb") as f:
            best_data = pickle.load(f)
        best_f1 = best_data["metrics"]["F1 Score"]
    except Exception:
        best_f1 = -1.0  # No best model exists yet

    if metrics["F1 Score"] > best_f1:
        with open(model_filename, "wb") as f:
            pickle.dump({"model": model, "metrics": metrics}, f)
        st.sidebar.success("Automatically saved new best model!")
    else:
        st.sidebar.info("Current model not better than the best saved model.")

def load_best_model(model_filename):
    """
    Attempts to load the best model from disk.
    Returns a tuple (model, metrics) if found, or (None, None) if no saved model exists.
    """
    try:
        with open(model_filename, "rb") as f:
            best_data = pickle.load(f)
        return best_data["model"], best_data["metrics"]
    except Exception:
        return None, None