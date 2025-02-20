import streamlit as st
from utils.model_trainer import get_trained_model
from utils.model_saver import auto_save_best_model, load_best_model

def main():
    st.title("Toxic Comment Detector")
    st.markdown("""
    This application is using manual implementation of Bagging Algorithm with Naive Bayer Classifier for toxicity detection.
    """)

    best_model, best_metrics = load_best_model("best_model.pkl")
    model, metrics = get_trained_model()
    
    if best_model is not None:
        if metrics["F1 Score"] > best_metrics["F1 Score"]:
            st.sidebar.info("Trained and saved better model than before.")
            auto_save_best_model(model, metrics)
        else:
            st.sidebar.info("Using best saved model.") 
            model, metrics = best_model, best_metrics
    else:
        st.sidebar.info("No best model found. Training new model.")
        auto_save_best_model(model, metrics)

    st.sidebar.header("Model Metrics")
    st.sidebar.write(f"**Accuracy:** {metrics['Accuracy'] * 100:.2f}%")
    st.sidebar.write(f"**Precision:** {metrics['Precision'] * 100:.2f}%")
    st.sidebar.write(f"**Recall:** {metrics['Recall'] * 100:.2f}%")
    st.sidebar.write(f"**F1 Score:** {metrics['F1 Score'] * 100:.2f}%")

    st.markdown("---")
    st.subheader("Check your comment for toxicity")
    comment = st.text_area("Write the comment you want to check:", height=100)
    if st.button("Evaluate"):
        if comment.strip() == "":
            st.error("Please write comment")
        else:
            prediction = model.predict([comment])
            if prediction[0] == 1 or prediction[0] == "1":
                st.error("Commect is **TOXIC** :rotating_light:")   
            else:
                st.success("Comment **is not toxic** :white_check_mark:")


if __name__ == '__main__':
    main()
