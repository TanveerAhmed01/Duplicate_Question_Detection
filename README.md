# Duplicate Question Detection 

## Overview
This project applies Natural Language Processing (NLP) and Machine Learning techniques to identify whether two questions have the same intent and are duplicates of each other. By extracting semantic, syntactic, and structural features from text pairs, the models classify question pairs as duplicates or non-duplicates.

## 📁 Dataset
The primary dataset used for this analysis is **`questions.csv`** (stored in a `Dataset` folder). 
It contains pairs of questions with the following key columns:
* **`question1` & `question2`:** The raw text of the question pairs.
* **`is_duplicate`:** The target variable (1 = Duplicate, 0 = Not a duplicate).

## Methodology
The project follows an advanced NLP data science pipeline:
1. **Data Preprocessing:** Cleaning text by removing HTML tags, expanding contractions, stripping punctuation, removing stopwords, and applying WordNet lemmatization.
2. **Feature Engineering:** A robust set of features was extracted to capture text similarity:
   * **Basic & Length Features:** Jaccard similarity, length differences, mean lengths, and longest common substring ratios.
   * **Fuzzy Features:** Ratios using the `FuzzyWuzzy` library (QRatio, partial ratio, token sort, token set).
   * **Semantic Embeddings:** Cosine similarity calculated using HuggingFace's pre-trained Sentence-BERT model (`all-MiniLM-L6-v2`).
   * **Syntactic & Keyword Features:** Part-of-Speech (POS) tagging overlap (nouns, verbs, adjectives) and RAKE (Rapid Automatic Keyword Extraction) overlaps.
3. **Model Building & Evaluation:** The data is split into training and testing sets, and multiple classifiers are trained and compared:
   * Logistic Regression
   * Random Forest Classifier
   * Gradient Boosting Classifier
   * XGBoost Classifier
   * LightGBM Classifier
4. **Visualization & Clustering:** Dimensionality reduction using t-SNE to visualize the feature space of the question pairs, and KMeans clustering evaluated with silhouette scores.

## Technologies Used
* **Python 3**
* **Pandas & NumPy:** Data manipulation
* **NLTK, FuzzyWuzzy & RAKE:** Text processing, string matching, and keyword extraction
* **Sentence-Transformers (HuggingFace):** Deep learning text embeddings
* **Scikit-Learn, XGBoost, & LightGBM:** Machine learning modeling and evaluation
* **Matplotlib:** Data visualization (t-SNE)

## How to Run
1. Clone this repository or download the files.
2. Ensure you have the required libraries installed (e.g., `transformers`, `sentence-transformers`, `fuzzywuzzy`, `xgboost`, `lightgbm`, `rake-nltk`).
3. Open `Duplicate_Question (1).ipynb` in Google Colab or Jupyter Notebook.
4. Upload the `questions.csv` dataset when prompted by the notebook.
5. Run all cells to process the text, extract features, and evaluate the classification models.
