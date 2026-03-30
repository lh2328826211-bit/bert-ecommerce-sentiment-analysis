Python 3.10.8 (tags/v3.10.8:aaaf517, Oct 11 2022, 16:50:30) [MSC v.1933 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import torch
... from transformers import BertTokenizer, BertForSequenceClassification
... 
... # ===================== 模型配置参数 =====================
... 
... MODEL_NAME = "bert-base-chinese"
... MAX_LEN = 128
... DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
... # 训练好的模型存放路径
... SAVE_MODEL_PATH = "./bert_emotion_model"
... 
... # ===================== 情感预测类 =====================
... class BertEmotionPredictor:
...     
...     def __init__(self):
...         # 加载 BERT 分词器
...         self.tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
...         # 加载本地训练好的模型
...         self.model = BertForSequenceClassification.from_pretrained(SAVE_MODEL_PATH)
...         # 将模型移动到指定设备
...         self.model.to(DEVICE)
...         # 切换为评估模式
...         self.model.eval()
... 
...     def preprocess(self, texts):
...         inputs = self.tokenizer(
...             texts,
...             padding="max_length",    # 填充到最大长度
...             truncation=True,         # 超长文本自动截断
...             max_length=MAX_LEN,      # 最大文本长度
...             return_tensors="pt"      # 返回 PyTorch 张量
...         )
...         return inputs
... 
...     @torch.no_grad()  # 关闭梯度计算、节省显存
...     def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        # 文本预处理
        inputs = self.preprocess(texts)
        # 将数据移动到运行设备
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # 模型推理
        outputs = self.model(**inputs)
        # 对输出 logits 进行 softmax 归一化，得到类别概率
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        # 获取概率最大的类别
        predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

        # 解析预测结果
        results = []
        for pred, prob in zip(predictions, probs):
            label = "正面" if pred == 1 else "负面"
            confidence = float(prob[pred])
            results.append({
                "label_id": int(pred),
                "label": label,
                "confidence": round(confidence, 4)
            })

        # 单条返回单个字典，多条返回列表
        return results[0] if len(results) == 1 else results


if __name__ == "__main__":
    predictor = BertEmotionPredictor()
    
    test_texts = [
        "预测的文本"
    ]
    
    # 批量预测
    for text in test_texts:
        res = predictor.predict(text)
        print(f"文本：{text}")
        print(f"情感：{res['label']}")
