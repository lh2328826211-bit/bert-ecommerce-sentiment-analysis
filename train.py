# 导入所需库
import re
import os
import jieba
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score,precision_score, recall_score




# ---------------------- 全局配置参数 ---------------------
  
# 数据集路径（建议上传时改为相对路径，如 "./data/online_shopping_10_cats.csv"）
DATA_PATH = "./data/online_shopping_10_cats.csv"
# 停用词路径（建议上传时改为相对路径，如 "./data/stopwords.txt"）
STOPWORDS_PATH = "./data/stopwords.txt"
NUM_LABELS = 2  # 分类数：二分类设为2
SAVE_MODEL_PATH = "./bert_emotion_model"  # 模型保存路径

MODEL_NAME = "bert-base-chinese"  # 使用的预训练 BERT 模型
MAX_LEN = 128  # 文本最大长度
BATCH_SIZE = 16  # 批次大小
LR = 2e-5  # 学习率
EPOCHS = 5  # 训练轮数
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 自动选择 GPU/CPU
print(f"当前训练设备: {DEVICE}")





 #1、 -------------------------------------------  数据预处理 ------------------------------------------  
  #加载停用词表
  def load_stopwords(stopword_path):
    stopwords = set()
    with open(stopword_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            word = line.strip()       # 去除每行首尾空白
            if word:                  # 跳过空行
                stopwords.add(word)
    return stopwords

#文本预处理
def preprocess_text(text, stopwords):
    if not isinstance(text, str):      # 处理非字符串类型（如 NaN）
        return ""
    text = re.sub(r"[^\u4e00-\u9fa5]", "", text)  
    word_list = jieba.lcut(text)    # 中文分词
    word_list = [w for w in word_list if w not in stopwords and len(w) > 1]  # 过滤停用词 + 长度小于2的词
    if len(word_list) < 2:      # 过滤太短的无效文本
        return ""
    return " " .join(word_list)               #  拼接为空格分隔的字符串


# 完整数据处理流程：加载数据 → 清洗 → 划分训练/验证集
def process_data(raw_data_path, stopwords_path):
    stopwords = load_stopwords(stopwords_path)
    df = pd.read_csv(raw_data_path, encoding="utf-8-sig")
    # 文本预处理，生成清洗后的列
    df["clean_text"] = df["review"].apply(lambda x: preprocess_text(x, stopwords))
    # 过滤空文本，重置索引
    df = df[df["clean_text"] != ""].reset_index(drop=True)
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    print(f"总样本：{len(df)} | 训练集：{len(train_df)} | 验证集：{len(val_df)}")
    return train_df, val_df




 #2、 ------------------------------------------- 构建数据加载器 ------------------------------------------
# 初始化 BERT 分词器
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
def bert_encoder(text, tokenizer, max_len):
    encoding = tokenizer.encode_plus(
        text=text,
        add_special_tokens=True,     # 添加 [CLS] 和 [SEP] 特殊标记
        max_length=max_len,
        padding="max_length",
        truncation=True,             # 超过长度则截断
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors="pt"           # 返回 PyTorch 张量
    )
    return encoding["input_ids"].flatten(), encoding["attention_mask"].flatten()          # 展平为一维张量，适配 DataLoader 输入

#自定义数据集类，用于加载情感分类数据
class TextEmotionDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts = df["clean_text"].tolist()   #
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    #返回数据集样本总数
    def __len__(self):
        return len(self.texts)
    #根据索引获取单条样本
    def __getitem__(self, idx):
        input_ids, attention_mask = bert_encoder(self.texts[idx], self.tokenizer, self.max_len)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label
        }

#构建数据加载器
def build_dataloader(train_df, val_df, tokenizer, max_len, batch_size):
    train_dataset = TextEmotionDataset(train_df, tokenizer, max_len)
    val_dataset = TextEmotionDataset(val_df, tokenizer, max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
...     return train_loader, val_loader
... 
... 
... #3、 ------------------------------------------- 构建模型与优化器 ------------------------------------------
... def init_model(model_name, num_labels, device):
...     model = BertForSequenceClassification.from_pretrained(
...         model_name,
...         num_labels=NUM_LABELS,  
...         problem_type="single_label_classification"    # 单标签分类
...     ).to(device)
...     return model
... 
... def init_optimizer(model, lr):
...     optimizer = torch.optim.AdamW(
...         model.parameters(),
...         lr=lr,
...         weight_decay=0.01     # L2 正则化，防止过拟合
...     )
...     return optimizer
... 
... 
... 
...   #4、 ------------------------------------------- 训练与验证 ------------------------------------------
...   def train_one_epoch(model, train_loader, optimizer, device, epoch, total_epochs):
...     model.train()     # 切换为训练模式
...     total_loss = 0.0
...     #进度条可视化
...     pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"训练: {epoch}/{total_epochs}")
...     for step, batch in pbar:
...         input_ids = batch["input_ids"].to(device)            # 将数据移动到指定设备
...         attention_mask = batch["attention_mask"].to(device)
...         labels = batch["label"].to(device)
... 
...         optimizer.zero_grad()       # 梯度清零
...         outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)   #前向传播
        loss = outputs.loss         # 计算损失
        loss.backward()             # 反向传播
        optimizer.step()            # 更新参数

        total_loss += loss.item()    
        pbar.set_postfix(loss=f"{loss.item():.4f}")    
    
    avg_loss = total_loss / len(train_loader)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():         # 关闭梯度计算，节省显存
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    train_acc = accuracy_score(all_labels, all_preds)
    return avg_loss, train_acc


def val_one_epoch(model, val_loader, device, epoch, total_epochs):
    model.eval()              # 切换为评估模式
    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"验证: {epoch}/{total_epochs}")
    with torch.no_grad():
        for step, batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")  #进度条显示损失
            
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="binary")       # 二分类精确率
    recall = recall_score(all_labels, all_preds, average="binary")             # 二分类召回率
    f1 = f1_score(all_labels, all_preds, average="binary")
    return avg_loss, acc, precision, recall, f1


# -------------------------------------------- 主函数 --------------------------------------------
if __name__ == "__main__":
    # 1. 数据预处理
    train_df, val_df = process_data(DATA_PATH, STOPWORDS_PATH)
    print(f"数据集大小 - 训练集: {len(train_df)}, 验证集: {len(val_df)}")
    print("数据预处理完成! \n" + "="*50)
    
    # 2. 构建数据加载器
    train_loader, val_loader = build_dataloader(train_df, val_df, tokenizer, MAX_LEN, BATCH_SIZE)
    
    # 3. 初始化模型和优化器
    model = init_model(MODEL_NAME, NUM_LABELS, DEVICE)
    optimizer = init_optimizer(model, LR)
    
    # 4. 模型训练
    best_acc = 0.0  # 记录最佳验证准确率
    print("开始模型训练\n" + "="*50)
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-"*50)
        # 训练
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, DEVICE, epoch+1, EPOCHS)
        # 验证
        val_loss, val_acc, val_precision, val_recall, val_f1 = val_one_epoch(model, val_loader, DEVICE, epoch+1, EPOCHS)
        
        # 打印日志
        print(f"训练集 - 损失: {train_loss:.4f}, 准确率: {train_acc:.4f}")
        print(f"验证集 - 损失: {val_loss:.4f}, 准确率: {val_acc:.4f}")
        print(f"精确率: {val_precision:.4f}, 召回率: {val_recall:.4f}, F1分数: {val_f1:.4f}")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            if not os.path.exists(SAVE_MODEL_PATH):
                os.makedirs(SAVE_MODEL_PATH)
            model.save_pretrained(SAVE_MODEL_PATH)  # 保存模型权重
            tokenizer.save_pretrained(SAVE_MODEL_PATH)  # 保存分词器
            print(f"当前模型为最佳模型、保存最佳模型, 准确率: {val_acc:.4f}")
