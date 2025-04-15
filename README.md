# 📰 Fake News Detection using LSTM

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Colab](https://img.shields.io/badge/Google%20Colab-Notebook-yellow?logo=google-colab)](https://colab.research.google.com/drive/1GYPvpXhklWbWDs6cHifhQO9rYp2nbkmt)

A deep learning-based fake news detector built using LSTM (Long Short-Term Memory) networks and pre-trained GloVe word embeddings.

---

## 📌 Problem Statement

In an age where misinformation spreads rapidly, detecting fake news is more critical than ever. This project aims to build a binary classification model that can accurately identify fake news articles based on their textual content.

---

## 🧠 Model Overview

- **Text Preprocessing**: Cleaning, tokenization, stopword removal
- **Word Embedding**: Pre-trained GloVe (100-dimensional vectors)
- **Model Architecture**:
  - Embedding Layer (non-trainable)
  - Dropout
  - 2x LSTM Layers
  - Dense Layers
  - Sigmoid Output Layer

---

## 📁 Dataset

The dataset includes:

- `text` - Full news article content
- `label` - 0 = Real, 1 = Fake

The following columns are dropped: `id`, `title`, `author`.

Dataset used: `train.csv`

---

## 🧹 Data Preprocessing

- Convert text to lowercase
- Remove punctuation and special characters
- Remove stopwords using NLTK
- Tokenize and pad sequences (max length = 500)

---

## 📊 Exploratory Data Analysis

Word clouds were generated to visualize frequently used words in:

- All news
- Real news (`label = 0`)
- Fake news (`label = 1`)

---

## 🔤 Word Embedding

Pre-trained GloVe embeddings (`glove.6B.100d.txt`) are used to convert words into semantic vectors. A custom embedding matrix is built to map vocabulary to embeddings.

---

## ⚙️ Model Training

- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam
- **Batch Size**: 128
- **Epochs**: 10
- **Validation Split**: 20%

Training and validation accuracy/loss are plotted after training.

---

## 📦 Dependencies

Install the required packages using:

```bash
pip install nltk wordcloud

## 🧪 How to Run This Project

Follow these steps to run the model:

---

### 🛠 1. Clone the Repository

```bash
git clone https://github.com/your-username/fake-news-detection-lstm.git
cd fake-news-detection-lstm
📦 2. Install Dependencies
Install dependencies locally:

bash
Copy
Edit
pip install -r requirements.txt
Or, if using Google Colab, install directly in a notebook cell:

python
Copy
Edit
!pip install nltk wordcloud
import nltk
nltk.download('stopwords')
📂 3. Prepare Dataset & Embeddings
Make sure the following files are in your project root or upload them in Colab when prompted:

train.csv — the labeled news dataset

glove.6B.100d.txt — pre-trained GloVe embeddings (100d)

Directory structure:

Copy
Edit
fake-news-detection-lstm/
├── train.csv
├── glove.6B.100d.txt
├── Fake_news_detection_LSTM.ipynb
└── README.md
🚀 4. Run the Notebook
Open the Jupyter notebook locally or in Colab and run all cells in order:

Open Fake_news_detection_LSTM.ipynb

Google Colab:
🔗 Open in Colab

📈 Output
✅ Word clouds for real and fake news

📊 Training vs. Validation Accuracy and Loss plots

🔍 Final model performance metrics

🛠️ Future Improvements
🔄 Use Bidirectional LSTM or GRU

🎯 Add an Attention mechanism

🌐 Deploy via Flask API or Streamlit

🤖 Experiment with Transformer models like BERT
