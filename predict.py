# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ===================== 1. 基础配置与预处理函数（与训练代码完全一致） =====================
# 下载必要的NLTK资源（首次运行需要）
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

# 固定随机种子
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# 设备配置：优先GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device for inference: {device}")

# 文本预处理函数（必须与训练时完全一致）
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    # 清洗文本：移除特殊字符、数字，转小写
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    text = text.lower().strip()
    # 分词 + 去停用词 + 词干提取
    tokens = nltk.word_tokenize(text)
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words and len(token) > 2]
    return tokens

# 加载训练时保存的词汇表（核心）
def load_vocab(vocab_path='vocab.txt'):
    """
    加载训练代码生成的vocab.txt
    :param vocab_path: 词汇表文件路径
    :return: 词汇表字典、PAD索引、UNK索引
    """
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file {vocab_path} not found! Please run training code first.")
    
    vocab = {}
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            token, idx = line.strip().split('\t')
            vocab[token] = int(idx)
    
    # 获取特殊标记的索引（训练时固定：<pad>=0, <unk>=1）
    pad_idx = vocab.get('<pad>', 0)
    unk_idx = vocab.get('<unk>', 1)
    
    print(f"✅ Vocabulary loaded from {vocab_path}, size: {len(vocab)}")
    print(f"  - <pad> index: {pad_idx}")
    print(f"  - <unk> index: {unk_idx}")
    return vocab, pad_idx, unk_idx

# 文本转模型输入张量（与训练时的处理逻辑一致）
def text_to_tensor(text, vocab, max_len=500):
    """
    将原始文本转为模型可识别的张量
    :param text: 输入英文文本
    :param vocab: 加载的词汇表
    :param max_len: 文本最大长度（需与训练时一致，默认500）
    :return: 模型输入张量 [1, max_len]、预处理后的tokens
    """
    # 预处理文本（和训练时相同的逻辑）
    tokens = preprocess_text(text)
    # 转索引序列（未知词映射为<unk>）
    indices = [vocab.get(token, vocab['<unk>']) for token in tokens]
    # 固定长度：补PAD/截断
    if len(indices) < max_len:
        indices += [vocab['<pad>']] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    # 转为张量并添加batch维度 [max_len] → [1, max_len]
    tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
    return tensor, tokens

# ===================== 2. 模型定义（与训练代码完全一致） =====================
# TextCNN模型
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=n_filters, 
                      kernel_size=(fs, embedding_dim)) for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text).unsqueeze(1)  # [batch, 1, seq_len, emb_dim]
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]  # [batch, n_filters, seq_len-fs+1]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]  # [batch, n_filters]
        cat = self.dropout(torch.cat(pooled, dim=1))  # [batch, n_filters * len(filter_sizes)]
        return torch.sigmoid(self.fc(cat))

# TextLSTM模型
class TextLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                            bidirectional=bidirectional, dropout=dropout if n_layers>1 else 0)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        text = text.permute(1, 0)  # [seq_len, batch_size]
        embedded = self.dropout(self.embedding(text))  # [seq_len, batch_size, emb_dim]
        output, (hidden, cell) = self.lstm(embedded)
        
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
        
        return torch.sigmoid(self.fc(hidden))

# ===================== 3. 模型加载与推理核心函数 =====================
def load_trained_model(model_type, vocab_size, model_path, pad_idx):
    """
    加载训练好的模型（超参数与训练代码完全一致）
    :param model_type: 'cnn' 或 'lstm'
    :param vocab_size: 词汇表大小（从加载的vocab获取）
    :param model_path: 模型权重文件路径（如TextCNN_best.pt）
    :param pad_idx: PAD标记的索引
    :return: 加载好的模型（eval模式）
    """
    # 超参数（必须与训练代码完全一致）
    EMBEDDING_DIM = 100
    OUTPUT_DIM = 1
    DROPOUT = 0.5

    # 加载对应模型
    if model_type.lower() == 'cnn':
        N_FILTERS = 100
        FILTER_SIZES = [3, 4, 5]
        model = TextCNN(
            vocab_size=vocab_size,
            embedding_dim=EMBEDDING_DIM,
            n_filters=N_FILTERS,
            filter_sizes=FILTER_SIZES,
            output_dim=OUTPUT_DIM,
            dropout=DROPOUT,
            pad_idx=pad_idx
        ).to(device)
    elif model_type.lower() == 'lstm':
        HIDDEN_DIM = 128
        N_LAYERS = 2
        BIDIRECTIONAL = True
        model = TextLSTM(
            vocab_size=vocab_size,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=OUTPUT_DIM,
            n_layers=N_LAYERS,
            bidirectional=BIDIRECTIONAL,
            dropout=DROPOUT,
            pad_idx=pad_idx
        ).to(device)
    else:
        raise ValueError("model_type must be 'cnn' or 'lstm'")

    # 加载模型权重（兼容CPU/GPU）
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found! Please check the path.")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 切换到评估模式（关闭Dropout）
    print(f"✅ {model_type.upper()} model loaded from {model_path}")
    return model

def predict_sentiment(text, model, vocab, max_len=500):
    """
    单文本情感预测核心函数
    :param text: 输入的英文文本
    :param model: 加载好的模型
    :param vocab: 加载的词汇表
    :param max_len: 文本最大长度（与训练一致）
    :return: 包含详细信息的预测结果字典
    """
    # 文本转张量
    tensor, tokens = text_to_tensor(text, vocab, max_len)
    
    # 推理（禁用梯度计算，提升速度）
    with torch.no_grad():
        pred = model(tensor).squeeze(0)  # [1,1] → [1]
        confidence = pred.item()  # 置信度（0~1，越接近1越正面，越接近0越负面）
        sentiment = "Positive" if confidence >= 0.5 else "Negative"
    
    # 返回结构化结果
    return {
        "input_text": text,
        "processed_tokens": tokens,  # 预处理后的tokens（便于调试）
        "sentiment": sentiment,      # 情感标签（Positive/Negative）
        "confidence": round(confidence, 4),  # 置信度（保留4位小数）
        "confidence_interpretation": f"{confidence*100:.2f}% {sentiment.lower()}"  # 可读性解释
    }

def batch_predict(texts, model, vocab, max_len=500):
    """
    批量文本情感预测
    :param texts: 文本列表
    :return: 预测结果列表
    """
    results = []
    print(f"\n📝 Starting batch prediction for {len(texts)} texts...")
    for i, text in enumerate(texts):
        result = predict_sentiment(text, model, vocab, max_len)
        result["sample_id"] = i + 1  # 添加样本ID，便于区分
        results.append(result)
        # 打印进度
        if (i + 1) % 5 == 0 or (i + 1) == len(texts):
            print(f"  - Processed {i + 1}/{len(texts)} samples")
    return results

# ===================== 4. 结果打印辅助函数 =====================
def print_prediction_result(result):
    """
    美观打印单条预测结果
    """
    print("\n" + "-"*80)
    print(f"Sample ID: {result.get('sample_id', 1)}")
    print(f"Input Text: {result['input_text'][:100]}..." if len(result['input_text'])>100 else f"Input Text: {result['input_text']}")
    print(f"Processed Tokens: {', '.join(result['processed_tokens'])[:100]}..." if len(result['processed_tokens'])>10 else f"Processed Tokens: {', '.join(result['processed_tokens'])}")
    print(f"Predicted Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence_interpretation']}")
    print("-"*80)

# ===================== 5. 主推理流程（可直接运行） =====================
if __name__ == '__main__':
    # -------------------------- 配置项（根据实际情况修改） --------------------------
    MODEL_TYPE = "cnn"  # 可选：'cnn' 或 'lstm'
    MODEL_PATH = "TextCNN_best.pt"  # 训练生成的模型权重路径
    VOCAB_PATH = "vocab.txt"  # 训练生成的词汇表路径
    MAX_LEN = 500  # 必须与训练时一致
    
    # -------------------------- 步骤1：加载词汇表 --------------------------
    try:
        vocab, pad_idx, unk_idx = load_vocab(VOCAB_PATH)
        vocab_size = len(vocab)
    except Exception as e:
        print(f"❌ Error loading vocabulary: {e}")
        exit()
    
    # -------------------------- 步骤2：加载模型 --------------------------
    try:
        model = load_trained_model(MODEL_TYPE, vocab_size, MODEL_PATH, pad_idx)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        exit()
    
    # -------------------------- 步骤3：单文本推理示例 --------------------------
    print("\n=== Single Text Inference Example ===")
    # 测试文本1（正面）
    test_text1 = "This movie is absolutely fantastic! The acting is brilliant and the plot is so engaging. I would watch it again and again."
    result1 = predict_sentiment(test_text1, model, vocab, MAX_LEN)
    print_prediction_result(result1)
    
    # 测试文本2（负面）
    test_text2 = "Worst movie I've ever seen! The story is boring, the characters are unlikable, and the ending is terrible. Total waste of time."
    result2 = predict_sentiment(test_text2, model, vocab, MAX_LEN)
    print_prediction_result(result2)
    
    # 测试文本3（中性偏正面）
    test_text3 = "The film was okay, not great but not terrible either. The cinematography was impressive and the soundtrack was nice."
    result3 = predict_sentiment(test_text3, model, vocab, MAX_LEN)
    print_prediction_result(result3)
    
    # -------------------------- 步骤4：批量推理示例（可选） --------------------------
    print("\n=== Batch Inference Example ===")
    batch_texts = [
        "Amazing cinematography and a touching story, highly recommended!",
        "I wasted 2 hours of my life on this garbage film.",
        "The movie had a slow start but the second half was really good.",
        "Terrible acting and a confusing plot, I regret watching it.",
        "One of the best movies I've seen this year, 10/10!"
    ]
    batch_results = batch_predict(batch_texts, model, vocab, MAX_LEN)
    
    # 打印批量结果
    for res in batch_results:
        print_prediction_result(res)
    
    # -------------------------- 最终提示 --------------------------
    print("\n🎉 Inference completed successfully!")
    print("💡 Tips:")
    print("  - Modify the 'test_text' or 'batch_texts' list to predict your own texts")
    print("  - Ensure MODEL_TYPE, MODEL_PATH, VOCAB_PATH match your training output")
    print("  - Confidence > 0.5 = Positive, < 0.5 = Negative")