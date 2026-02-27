# A-Simple-Sentiment-Analysis
Code for movie reviews sentiment analysis
This repository implements binary sentiment classification (positive/negative) on the IMDB movie review dataset using TextCNN and TextLSTM, with complete data preprocessing, model training, evaluation, visualization, and inference pipelines.

**Quick Start**

**1. Environment Setup**
Install required dependencies:

pip install torch nltk scikit-learn matplotlib seaborn

**2. Data Preprocessing**
The raw IMDB dataset is automatically downloaded and processed through the following steps:
Text Cleaning: Remove non-alphabetic characters and convert text to lowercase.
Tokenization: Split text into word tokens with NLTK.
Stopword Removal: Filter out low-semantic stopwords (e.g., "the", "is").
Stemming: Unify word forms via Porter Stemmer (e.g., "fantastic" → "fantast").
Vocabulary Construction: Build a 25,000-size vocabulary from the training set (with <pad> and <unk> tokens) and save it as vocab.txt.
Sequence Normalization: Truncate/pad all text sequences to a fixed length of 500 tokens for model input.

**3. Model Training**
Run the training script to:
Train TextCNN and TextLSTM models (10 epochs, batch size=64, Adam optimizer).
Save the best model weights (based on validation loss) as TextCNN_best.pt/TextLSTM_best.pt.
Generate visualization plots (word frequency, text length distribution, training curves, etc.).

**4. Inference**
Use the inference script to:
Load the saved vocab.txt and model weights.
Predict sentiment (Positive/Negative) for single/batch English movie reviews with confidence scores.

