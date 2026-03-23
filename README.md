# 🎬 A-Simple-Sentiment-Analysis
This repository implements binary sentiment classification (positive/negative) on the IMDB movie review dataset using **TextCNN** and **TextLSTM** neural networks. It provides end-to-end pipelines for data preprocessing, model training, evaluation, visualization, and inference, designed for reproducible and educational purposes.

---

## ✨ Overview
This project focuses on building and evaluating neural network models for movie review sentiment analysis. Two classic deep learning architectures are implemented and compared:
- **TextCNN** 🧇: Convolutional neural network tailored for text data, effective at capturing local semantic features (e.g., emotion-laden keywords).
- **TextLSTM** 🌀: Long Short-Term Memory network, specialized in modeling sequential dependencies and contextual information in text.

---

## 🚀 Quick Start
### 1. 🍪 Environment Setup
Install the required dependencies to run the code:
```bash
pip install torch nltk scikit-learn matplotlib seaborn
```
> 💡 Note: For installation issues, upgrade the relevant package with `pip install --upgrade [package name]` to ensure compatibility.

### 2. 🧹 Data Preprocessing
The raw IMDB dataset is automatically downloaded and processed through the following standardized steps:

| Step | Description | Implementation Details |
|------|-------------|------------------------|
| 🧽 Text Cleaning | Remove non-alphabetic characters and normalize text to lowercase to reduce noise | Filter out non-a-z characters; convert all text to lowercase |
| ✂️ Tokenization | Split text into discrete word tokens for model input | Tokenize text using NLTK's word tokenizer |
| 🗑️ Stopword Removal | Eliminate low-information stopwords to focus on semantically meaningful terms | Filter out stopwords (e.g., "the", "is", "a") from the NLTK stopword list |
| 🔨 Stemming | Unify word forms to reduce vocabulary size and improve generalization | Apply Porter Stemmer to standardize word morphology (e.g., "fantastic" → "fantast") |
| 📚 Vocabulary Construction | Build a fixed-size vocabulary from the training set to map words to indices | Create a 25,000-word vocabulary with `<PAD>` (padding) and `<UNK>` (unknown token) ; save as `vocab.txt` |
| 📏 Sequence Normalization | Standardize input sequence length for consistent model input | Truncate/pad all text sequences to a fixed length of 500 tokens |

### 3. 🏋️ Model Training
Execute the training script to train and validate the TextCNN and TextLSTM models:
```bash
python train.py
```
#### Training Details:
- 🎯 Train TextCNN and TextLSTM for 10 epochs with a batch size of 64
- 📦 Optimize model parameters using the Adam optimizer
- 🏆 Save the best model weights (based on validation loss) as `TextCNN_best.pt`/`TextLSTM_best.pt`
- 📊 Generate quantitative and qualitative visualization results:
  - Word frequency distribution (key terms in positive/negative reviews)
  - Text length distribution of the IMDB dataset
  - Training curves (loss and accuracy over epochs)

### 4. 🎯 Inference
Use the trained models to predict sentiment for new movie reviews:
```bash
python predict.py
```
#### Inference Capabilities:
- 📖 Load pre-saved `vocab.txt` and model weights for inference
- 🗣️ Predict sentiment (Positive/Negative) for single movie reviews
- 📥 Batch inference for multiple reviews
- 📊 Output confidence scores for each prediction (reflecting model certainty)

---

## 📊 Visualization
Comprehensive visualization is provided to analyze data characteristics and model performance:
1. **Word Frequency Cloud** ☁️: Visualize the most frequent words in positive/negative reviews (red = negative, green = positive)
2. **Text Length Histogram** 📊: Distribution of review lengths in the IMDB dataset (majority range: 100–500 tokens)
3. **Training Loss/Accuracy Curves** 📈: Track model convergence over training epochs (decreasing loss, increasing accuracy)
4. **Confusion Matrix** 🧮: Evaluate classification performance by quantifying true/false positives/negatives

---

## 🐾 Example Outputs
### Input Review:
> "This movie made me cry happy tears—the characters were so lovable and the plot was perfect! I watched it 3 times!"

### Output:
> Sentiment: Positive 😊 | Confidence: 99.2%

### Input Review:
> "Worst movie ever—the acting was terrible and the plot made no sense. I left the theater early!"

### Output:
> Sentiment: Negative 😞 | Confidence: 98.7%

---

## 🎨 Model Characteristics
- 🧇 **TextCNN**: Excels at extracting local n-gram features and key emotional keywords, with fast training and inference speed.
- 🌀 **TextLSTM**: Superior at capturing long-range contextual dependencies (e.g., negation: "not good" ≠ "good"), with stronger sequential understanding.
- 📊 Both models achieve approximately 85% classification accuracy on the IMDB dataset, demonstrating strong performance for beginner-friendly implementations.

---

## ✨ License
This code is released for educational and academic use, suitable for coursework, self-learning, and research purposes.
