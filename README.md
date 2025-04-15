📰 Fake News Detection using LSTM
This project focuses on building a deep learning model using LSTM (Long Short-Term Memory) networks to detect whether a news article is real or fake. The solution leverages Natural Language Processing (NLP) techniques, pre-trained GloVe word embeddings, and a stacked LSTM architecture to classify news articles based on their content.

🚀 Overview
Fake news has become a major issue in the digital age. This project aims to build a robust text classification model that can accurately identify fake news using deep learning.

The dataset is preprocessed to clean and tokenize the text, and pre-trained GloVe embeddings are used to provide semantic meaning. The model is trained on this data using a two-layer LSTM followed by fully connected layers for classification.

🧠 Model Architecture
Embedding Layer using pre-trained GloVe vectors

Dropout to reduce overfitting

Stacked LSTM Layers for sequence learning

Dense Layers for final classification

Sigmoid Activation for binary output (Fake or Real)

📁 Dataset
The dataset contains labeled news articles with columns like:

id (Dropped)

title (Dropped)

author (Dropped)

text (Used)

label (Target: 0 = Real, 1 = Fake)

Dataset used: train.csv

🧹 Preprocessing
Lowercasing text

Removing punctuation and newline characters

Removing stopwords

Tokenization

Padding sequences for uniform input size

📊 Exploratory Data Analysis
Word clouds of overall, real, and fake news content to visualize frequent words

🔤 Embedding
Tokenizer converts words to sequences

Pre-trained GloVe embeddings (glove.6B.100d.txt) used to create an embedding matrix

Embedding layer initialized with GloVe weights

⚙️ Model Training
Binary classification using binary_crossentropy loss

Optimizer: Adam

10 epochs

Batch size: 128

Stratified train-test split for balanced class distribution

📈 Evaluation
Plots included:

Training vs Validation Accuracy

Training vs Validation Loss

These metrics help evaluate model performance over epochs.

📦 Dependencies
Install the required libraries using pip:

bash
Copy
Edit
pip install wordcloud nltk
Also, download NLTK stopwords using:

python
Copy
Edit
import nltk
nltk.download('stopwords')
🧪 Usage
To run this notebook:

Upload train.csv and glove.6B.100d.txt to your Colab environment.

Run all cells in order.

Review the plots to evaluate model performance.

📌 Notes
This project is run on Google Colab.

Large GloVe embeddings file is required (100d version used).

Focus is on proof-of-concept, not production deployment.

