# Toxic Comment Detector

This project is an implementation of a basic text classifier to detect toxic comments and specific subcategories such as provocative or abusive language. It uses a custom implementation of a Bagging Classifier on top of a simple Naive Bayes model and evaluation functions.

## Overview

The project:
- Loads text data and associated binary labels (e.g. IsToxic, IsProvocative, IsAbusive) from CSV.
- Uses a simple bag-of-words Naive Bayes classifier with Laplace smoothing as the base estimator.
- Implements bagging (bootstrap aggregation) to combine multiple estimators for improved performance.
- Evaluates models using standard metrics — accuracy, precision, recall, and F1 score.
- Uses Streamlit to provide a simple web interface for testing single comments.
- Comments in code is written on Slovak language.

## Architecture

1. **Data Loading**  
   - **File:** `utils/data_loader.py`  
   - **Description:** Loads data from the CSV file (`data/youtoxic_english_1000.csv`) and extracts the text and labels for a specified category.

2. **Base Classification Model**  
   - **File:** `algorithm/naive_bayes.py`  
   - **Description:** Implements a simple Naive Bayes classifier that:

     - Tokenizes input text.
     - Counts words per class and calculates class priors.
     - Predicts class labels based on maximum likelihood using Laplace smoothing.

3. **Bagging Aggregation**  
   - **File:** `algorithm/bagging.py`  
   - **Description:** Implements the Bagging algorithm that:
   
     - Creates multiple bootstrap samples of the data.
     - Trains a separate base estimator (SimpleNaiveBayesClassifier) on each sample.
     - Aggregates predictions using majority voting.

4. **Model Training**  
   - **File:** `utils/model_trainer.py`  
   - **Description:**  
     - Provides a function `train_model` that shuffles data, splits it into training, test and evaluation sets (70/15/15), trains the Bagging classifier, and calculates evaluation metrics on the test set.
     - The `get_trained_model` function loads the data for a given label and returns the model and its metrics.

5. **Evaluation Metrics**  
   - **File:** `metrics/evaluation.py`  
   - **Description:**  
     - Contains helper functions to compute accuracy, precision, recall, and F1 score for model predictions.

6. **Model Saving and Loading**  
   - **File:** `utils/model_saver.py`  
   - **Description:**  
     - Provides functions to automatically save the models if it outperforms the previous best (using F1 score) and load the best model. All models is stored in `models` directory.

7. **Application Interface**  
   - **File:** `App.py`  
   - **Description:**  
     - Uses Streamlit to create a web interface where users can:
       - Trigger model training.
       - View calculated metrics for each label.
       - Evaluate a new comment across the models (IsToxic, IsProvocative, IsAbusive).

## Running the Application

1. **Dependencies**  
   Ensure you have Python installed as well as necessary packages such as:
   - pandas
   - streamlit

   For installing dependencies:
   ```bash
   pip install -r requirements.txt

2. **Data**

   My program is using 4 attributes in dataset: Text, IsToxic, IsAbusive, IsProvocative. If you want to use your dataset put it in `data` and make sure there is same columns or change code in `data_loader.py`.

3. **Run the App**
    ```bash
    streamlit run App.py

## Documentation
If you want more detailed description, you can read documentation that also is in this repository named `TCD_docs`.
