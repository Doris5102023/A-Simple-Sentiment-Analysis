# 🎬 A-Simple-Sentiment-Analysis 🥰
Easy code for movie reviews sentiment analysis! This repo uses **TextCNN** and **TextLSTM** (two super neural networks 🧠) to do binary sentiment classification (positive 😊/negative 😞) on the IMDB movie review dataset. We've got full pipelines for data preprocessing, model training, evaluation, visualization, and inference—all wrapped in a fluffy blanket of cuteness 🧶!

---

## ✨ What's in This Repo?
Imagine teaching a little AI bear 🐻 to read movie reviews and tell if people loved them (👍) or hated them (👎)—that's exactly what we're doing here!  
We use two tiny but powerful models:
- **TextCNN** 🧇: Like a cookie cutter that picks out "tasty" keywords (e.g., "amazing", "terrible") from reviews
- **TextLSTM** 🌀: Like a storybook reader that remembers the flow of words (e.g., "not good" ≠ "good"!)

---

## 🚀 Quick Start (Let's Play with the AI Bear! 🐾)
### 1. 🍪 Environment Setup (Feed the AI Bear Snacks!)
First, install all the "snacks" (dependencies) the AI bear needs to work:
```bash
pip install torch nltk scikit-learn matplotlib seaborn
```
> 💡 Pro Tip: If the bear is picky (installation errors), use `pip install --upgrade [package name]` to give it fresh snacks!

### 2. 🧹 Data Preprocessing (Clean Up the Movie Reviews!)
The raw IMDB dataset is downloaded automatically—we just need to "tidy up" the reviews like folding tiny clothes 👕:

| Step | Cute Explanation | What We Actually Do |
|------|------------------|---------------------|
| 🧽 Text Cleaning | Scrub off yucky non-alphabet characters (like `!@#$%`) and make all words lowercase (e.g., "Amazing" → "amazing") | Remove non-a-z chars, lowercase text |
| ✂️ Tokenization | Cut reviews into tiny word pieces (e.g., "I love this movie" → ["I", "love", "this", "movie"]) | Split text with NLTK tokenizer |
| 🗑️ Stopword Removal | Throw away boring words that mean nothing (e.g., "the", "is", "a")—like picking out crumbs from a cookie 🍪 | Filter stopwords (e.g., "the", "and") |
| 🔨 Stemming | Make words wear the same "uniform" (e.g., "fantastic" → "fantast", "running" → "run") | Unify word forms with Porter Stemmer |
| 📚 Vocabulary Construction | Build a tiny dictionary (25,000 words!) for the AI bear to learn—add special tokens `<PAD>` (blank space 📝) and `<UNK>` (unknown word ❓) | Create 25k vocab from training set, save as `vocab.txt` |
| 📏 Sequence Normalization | Cut long reviews short and pad short reviews (all to 500 words!)—like cutting cake into equal slices 🎂 | Truncate/pad sequences to 500 tokens |

### 3. 🏋️ Model Training (Train the AI Bear!)
Run the training script to teach the AI bear to read reviews—watch it get smarter every epoch! 📈
```bash
# Just run the training script (we've set all the cute hyperparameters for you!)
python train.py
```
#### What Happens During Training?
- 🎯 We train **TextCNN** and **TextLSTM** for 10 rounds (epochs)
- 📦 Feed the bear 64 reviews at a time (batch size=64) with the Adam optimizer (its favorite trainer 🏋️)
- 🏆 Save the best model weights (based on validation loss) as `TextCNN_best.pt`/`TextLSTM_best.pt` (golden medals 🥇 for the bear!)
- 📊 Generate super plots:
  - Word frequency (which words are most common in positive/negative reviews 📝)
  - Text length distribution (how long are most reviews? 📏)
  - Training curves (see the bear's accuracy go up! 📈)

### 4. 🎯 Inference (Let the AI Bear Guess Sentiment!)
Now the AI bear is trained—let it read new reviews and tell you if they're positive/negative! 🐻
```bash
python predict.py
```
#### What You Can Do:
- 📖 Load the saved `vocab.txt` (the bear's dictionary) and model weights (its trained brain 🧠)
- 🗣️ Predict sentiment for **single reviews** (e.g., "This movie was so cute and heartwarming! 🥹")
- 📥 Predict sentiment for **batch reviews** (feed the bear a bunch of reviews at once!)
- 📊 Get confidence scores (how sure the bear is—e.g., 98% positive 😊)

---

## 📊 Visualization (Plots Galore! 📈)
We've got tons of fluffy visualizations to show off the AI bear's skills:
1. **Word Frequency Cloud** ☁️: Colorful clouds of words (red=negative 😡, green=positive 🥰)
2. **Text Length Histogram** 📊: How long are IMDB reviews? (most are 100-500 words!)
3. **Training Loss/Accuracy Curves** 📈: Watch the bear get better at guessing—loss goes down, accuracy goes up!
4. **Confusion Matrix** 🧮: See how many reviews the bear guessed right/wrong (we promise it's better than a human! 🧑‍🦰→🐻)

---

## 🐾 Example Outputs (What the Bear Says!)
### Input Review:
> "This movie made me cry happy tears 🥹—the characters were so lovable and the plot was perfect! I watched it 3 times!"

### Output:
> Sentiment: Positive 😊 | Confidence: 99.2%

### Input Review:
> "Worst movie ever 🤢—the acting was terrible and the plot made no sense. I left the theater early!"

### Output:
> Sentiment: Negative 😞 | Confidence: 98.7%

---

## 🎨 Fun Facts About the Models
- 🧇 **TextCNN**: Great at picking out "key emotions" (e.g., "amazing", "horrible")—like a bear sniffing out honey 🍯
- 🌀 **TextLSTM**: Great at understanding context (e.g., "not good" = bad, "not bad" = good)—like a bear reading a storybook 📖
- 🐻 Both models are tiny (perfect for beginners!) but get ~85% accuracy on IMDB—way better than guessing!

---

## ✨ License
This code is as free as a bear playing in the forest 🌳—use it for learning, homework, or just for fun!
