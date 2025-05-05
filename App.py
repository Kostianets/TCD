import streamlit as st
from utils.model_trainer import get_trained_model
from utils.model_saver import auto_save_best_model, load_best_model
import re

def model_loader(best_model, best_metrics, model, metrics, label: str):
    """
    Načíta model zo súboru, ak je lepší ako aktuálne natrenovany model.

    Parameters
    ----------
    - best_model: natrénovaný model
    - best_metrics: slovník metrík modelu
    - model: natrénovaný model
    - metrics: slovník metrík modelu
    - label: názov stĺpca, ktorý obsahuje labely

    Returns
    -------
    - model: natrénovaný model
    - metrics: slovník metrík model
    """
    if best_model is not None:
        if metrics["F1 Score"] > best_metrics["F1 Score"]:
            st.sidebar.info(f"Trained and saved better model than before for {label}.")
            auto_save_best_model(model, metrics, f"models/best_model_{label}.pkl")
        else:
            model, metrics = best_model, best_metrics
            #metrics.update({"Accuracy": 0.0, "Precision": 0.0, "Recall": 0.0, "F1 Score": 0.0})
    else: 
        st.sidebar.info(f"No best model found for {label}. Training new model.")
        auto_save_best_model(model, metrics, f"models/best_model_{label}.pkl")
    return model, metrics

def metrics(metrics, label: str):
    """
    Vypíše metriky modelu.
    """
    print(f"**Accuracy for {label}:** {metrics['Accuracy'] * 100:.2f}%")
    print(f"**Precision for {label}:** {metrics['Precision'] * 100:.2f}%")
    print(f"**Recall for {label}:** {metrics['Recall'] * 100:.2f}%")
    print(f"**F1 Score for {label}:** {metrics['F1 Score'] * 100:.2f}%\n")

def highlight_words(text, words_to_highlight):
    """
    Highlights specified words in the text using HTML styling.

    Parameters
    ----------
    - text: input text (comment)
    - words_to_highlight: list of words to highlight

    Returns
    -------
    - str: HTML string with highlighted words
    """
    words_to_highlight = set(w.lower() for w in words_to_highlight)
    def clean_word(w):
        return re.sub(r'[^\w]', '', w).lower()
    highlighted = []
    for word in text.split():
        if clean_word(word) in words_to_highlight:
            highlighted.append(f'<span class="highlight">{word}</span>')
        else:
            highlighted.append(word)
    return ' '.join(highlighted)

def main():
    css = """
    <style>
    /* App background, font and heading colors */
    .stApp {
        background-color: #1E1E2E;
        color: #E0E0E0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        padding: 20px;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #4FD1C5;          /* mint accent */
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.7);
        margin-bottom: 0.5em;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* Sidebar styling */
    .stSidebar {
        background-color: #2A2A3E;
        padding: 20px;
        border-right: 2px solid #44475a;
    }
    /* Only style buttons in the sidebar */
    .stSidebar .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border-radius: 12px;
        padding: 8px 16px;
        font-size: 14px;
        font-weight: 600;
        transition: background 0.3s ease, transform 0.1s ease;
    }
    .stSidebar .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2, #667eea);
        transform: translateY(-1px);
    }

    /* Text area and text input */
    .stTextArea textarea, .stTextInput input {
        background-color: #282A36;
        color: #F8F8F2;
        border: 1px solid #6272a4;
        border-radius: 8px;
        padding: 12px;
        font-size: 15px;
        width: 100%;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #50FA7B;
        box-shadow: 0 0 6px rgba(80, 250, 123, 0.5);
        outline: none;
    }

    /* Section separators */
    hr, .stMarkdown hr {
        border: none;
        border-top: 1px solid #44475a;
        margin: 2em 0;
    }

    /* Success and error messages */
    .stSuccess, .stError {
        background-color: #282A36;
        color: #F8F8F2;
        border-radius: 6px;
        padding: 12px;
        margin: 12px 0;
        border-left: 4px solid;
    }
    .stSuccess {
        border-color: #50FA7B;
    }
    .stError {
        border-color: #FF5555;
    }

    /* Responsive tweaks */
    @media (max-width: 768px) {
        .stApp {
            padding: 10px;
        }
        .stTextArea textarea, .stTextInput input {
            font-size: 14px;
            padding: 10px;
        }
    }
    </style>
    """
    
    st.markdown(css, unsafe_allow_html=True)
    
    st.sidebar.title("Toxic Comment Detector")
    st.sidebar.markdown("""
    This application is using manual implementation of Bagging Algorithm with Naive Bayes Classifier for toxicity, abusing and provocation detection.
    """)

    best_model_Toxic, best_metrics_Toxic = load_best_model("models/best_model_IsToxic.pkl")
    modelToxic, metricsToxic = get_trained_model("IsToxic")

    best_model_Provocative, best_metrics_Provocative = load_best_model("models/best_model_IsProvocative.pkl")
    modelProvocative, metricsProvocative = get_trained_model("IsProvocative")

    best_model_Abusive, best_metrics_Abusive = load_best_model("models/best_model_IsAbusive.pkl")
    modelAbusive, metricsAbusive = get_trained_model("IsAbusive")
    
    modelToxic, metricsToxic = model_loader(best_model_Toxic, best_metrics_Toxic, modelToxic, metricsToxic, "IsToxic")
    modelProvocative, metricsProvocative = model_loader(best_model_Provocative, best_metrics_Provocative, modelProvocative, metricsProvocative, "IsProvocative")
    modelAbusive, metricsAbusive = model_loader(best_model_Abusive, best_metrics_Abusive, modelAbusive, metricsAbusive, "IsAbusive")

    #st.sidebar.header("Model Metrics")
    metrics(metricsToxic, "IsToxic")
    metrics(metricsProvocative, "IsProvocative")
    metrics(metricsAbusive, "IsAbusive")
    print("-----------------------------------")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Check your comment")
    comment = st.sidebar.text_area("Write the comment you want to check:", height=100)
    if st.sidebar.button("Evaluate"):
        if comment.strip() == "":
            st.error("Please write comment")
        else:
            predictionToxic = modelToxic.predict([comment])[0]
            predictionProvocative = modelProvocative.predict([comment])[0]
            predictionAbusive = modelAbusive.predict([comment])[0]

            st.write("## Comment Analysis")
            
            if predictionToxic == 1 or predictionAbusive == 1 or predictionProvocative == 1:
                st.write("The comment has been classified as:")
                if predictionToxic == 1 or predictionToxic == "1":
                    st.write("- TOXIC")
                if predictionAbusive == 1 or predictionAbusive == "1":
                    st.write("- ABUSIVE")
                if predictionProvocative == 1 or predictionProvocative == "1":
                    st.write("- PROVOCATIVE")
            else:
                st.write("The comment is not toxic, abusive, or provocative.")
            
            if predictionToxic == 1 or predictionToxic == "1":
                st.write("### Toxicity Analysis")
                toxic_words = modelToxic.get_contributing_words(comment)
                highlighted_comment = highlight_words(comment, toxic_words)
                st.write("The comment is **TOXIC**")
                #st.markdown(highlighted_comment, unsafe_allow_html=True)
                st.write("Contributing words:", ", ".join(toxic_words))

            if predictionAbusive == 1 or predictionAbusive == "1":
                st.write("### Abusiveness Analysis")
                abusive_words = modelAbusive.get_contributing_words(comment)
                highlighted_comment = highlight_words(comment, abusive_words)
                st.write("The comment is **ABUSIVE**")
                #st.markdown(highlighted_comment, unsafe_allow_html=True)
                st.write("Contributing words:", ", ".join(abusive_words))

            if predictionProvocative == 1 or predictionProvocative == "1":
                st.write("### Provocativeness Analysis")
                provocative_words = modelProvocative.get_contributing_words(comment)
                highlighted_comment = highlight_words(comment, provocative_words)
                st.write("The comment is **PROVOCATIVE**")
                #st.markdown(highlighted_comment, unsafe_allow_html=True)
                st.write("Contributing words:", ", ".join(provocative_words))

            #if predictionToxic[0] == 1 or predictionToxic[0] == "1":
            #    st.error("Commect is **TOXIC** :rotating_light:")   
            #else:
            #    st.success("Comment **is not toxic** :white_check_mark:")
            #if predictionAbusive[0] == 1 or predictionAbusive[0] == "1":
            #    st.error("Commect is **ABUSIVE** :rotating_light:")
            #if predictionProvocative[0] == 1 or predictionProvocative[0] == "1":
            #    st.error("Commect is **PROVOCATIVE** :rotating_light:")


if __name__ == '__main__':
    main()
