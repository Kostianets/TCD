import streamlit as st
from utils.model_trainer import get_trained_model
from utils.model_saver import auto_save_best_model, load_best_model

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

def main():
    css = """
    <style>
    /* General styling for the Streamlit app */
    .stApp {
        background-color: #1A1A1A; /* Dark background */
        color: white; /* White text */
        font-family: Arial, sans-serif; /* Clean font */
    }

    /* Container for centering content */
    .container {
        max-width: 800px; /* Constrain width */
        margin: 0 auto; /* Center container */
        padding: 20px; /* Internal spacing */
    }

    /* Header styling */
    header {
        border-bottom: 1px solid #444; /* Subtle divider */
        padding-bottom: 20px; /* Space below header */
    }

    header h1 {
        font-size: 2.8em; /* Large title */
        text-transform: uppercase; /* Uppercase text */
        text-align: center; /* Centered */
        margin-bottom: 10px; /* Space below title */
    }

    header p {
        font-size: 1.1em; /* Subtitle size */
        text-align: center; /* Centered */
        line-height: 1.5; /* Improved readability */
    }

    /* Main content styling */
    main h2 {
        font-size: 2em; /* Bold subheading */
        text-align: center; /* Centered */
        margin-top: 30px; /* Space above */
    }

    main p {
        font-size: 1em; /* Regular text */
        text-align: center; /* Centered */
        margin-bottom: 20px; /* Space below */
    }

    /* Textarea styling for comment input */
    .stTextArea textarea {
        width: 100%; /* Full width */
        height: 150px; /* Tall input area */
        background-color: #333333; /* Dark gray background */
        color: white; /* White text */
        border: 1px solid #444; /* Subtle border */
        border-radius: 10px; /* Rounded corners */
        padding: 10px; /* Internal spacing */
        font-size: 1em; /* Consistent font size */
        resize: none; /* Prevent resizing */
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.3); /* Subtle shadow */
    }

    .stTextArea textarea::placeholder {
        color: #aaa; /* Light gray placeholder */
    }

    .stTextArea textarea:focus {
        outline: none; /* Remove default outline */
        border-color: #28A745; /* Green border on focus */
    }

    /* Button styling */
    .stButton>button {
        display: block; /* Block for centering */
        margin: 20px auto; /* Center button */
        background-color: #28A745; /* Green background */
        color: white; /* White text */
        border: none; /* No border */
        border-radius: 5px; /* Rounded corners */
        padding: 10px 20px; /* Comfortable padding */
        font-size: 1em; /* Consistent font size */
        cursor: pointer; /* Hand cursor on hover */
        min-width: 120px; /* Ensure button width */
        transition: background-color 0.3s ease; /* Smooth hover transition */
    }

    .stButton>button:hover {
        background-color: #218838; /* Darker green on hover */
    }

    /* Responsive adjustments */
    @media (max-width: 600px) {
        header h1 {
            font-size: 2em; /* Smaller title on small screens */
        }
        main h2 {
            font-size: 1.5em; /* Smaller subheading */
        }
        .stTextArea textarea {
            height: 100px; /* Reduced height for smaller screens */
        }
    }

    @media (max-width: 400px) {
        .container Quartz {
            padding: 10px; /* Reduced padding on very small screens */
        }
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

    st.title("Toxic Comment Detector")
    st.markdown("""
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

    st.markdown("---")
    st.subheader("Check your comment")
    comment = st.text_area("Write the comment you want to check:", height=100)
    if st.button("Evaluate"):
        if comment.strip() == "":
            st.error("Please write comment")
        else:
            predictionToxic = modelToxic.predict([comment])
            predictionProvocative = modelProvocative.predict([comment])
            predictionAbusive = modelAbusive.predict([comment])
            if predictionToxic[0] == 1 or predictionToxic[0] == "1":
                st.error("Commect is **TOXIC** :rotating_light:")   
            else:
                st.success("Comment **is not toxic** :white_check_mark:")
            if predictionAbusive[0] == 1 or predictionAbusive[0] == "1":
                st.error("Commect is **ABUSIVE** :rotating_light:")
            if predictionProvocative[0] == 1 or predictionProvocative[0] == "1":
                st.error("Commect is **PROVOCATIVE** :rotating_light:")


if __name__ == '__main__':
    main()
