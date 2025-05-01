import os
import multiprocessing
import numpy as np
import pandas as pd
from tqdm import tqdm
import jieba 

from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import seaborn as sns

# 文件路径
TRAIN_PATH = 'D:/Users/CN096/NLP/word2vec/train_data.csv'
TEST_PATH = 'D:/Users/CN096/NLP/word2vec/test_data.csv'
STOPWORDS_PATH = 'D:/Users/CN096/NLP/word2vec/stopwords.txt'

# 加载数据
# 读取 Excel 文件
train_df = pd.read_excel(TRAIN_PATH)
test_df = pd.read_excel(TEST_PATH)

print("训练样本数:", len(train_df), "测试样本数:", len(test_df))
labels = sorted(train_df['label'].unique())

# 加载中文停用词
with open(STOPWORDS_PATH, 'r', encoding='utf-8') as f:
    stop_words = set(line.strip() for line in f)

# 中文分词 + 清洗
def clean_text(text):
    tokens = jieba.lcut(str(text))
    tokens = [w for w in tokens if w.strip() and w not in stop_words]
    return tokens

tqdm.pandas()
train_df['tokens'] = train_df['text'].progress_apply(clean_text)
test_df['tokens'] = test_df['text'].progress_apply(clean_text)

# 用训练集训练 Word2Vec
sentences = train_df['tokens'].tolist()
EMBEDDINGS = {}

for dim in [100, 300]:
    print(f"Training Word2Vec {dim}-dim...")
    w2v = Word2Vec(sentences=sentences,
                   vector_size=dim,
                   window=5,
                   min_count=2,
                   workers=multiprocessing.cpu_count(),
                   epochs=10)
    EMBEDDINGS[dim] = w2v

# 向量生成方法
def doc_vector_mean(tokens, model):
    vecs = [model.wv[w] for w in tokens if w in model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size)

def doc_vector_tfidf(tokens, model, tfidf_dict):
    vecs = []
    weights = []
    for w in tokens:
        if w in model.wv and w in tfidf_dict:
            vecs.append(model.wv[w] * tfidf_dict[w])
            weights.append(tfidf_dict[w])
    if not vecs:
        return np.zeros(model.vector_size)
    return np.sum(vecs, axis=0) / np.sum(weights)

# 构建 TF-IDF 词典（基于训练集）
tfidf = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x, token_pattern=None, min_df=2)
tfidf_matrix = tfidf.fit_transform(train_df['tokens'])
idf_scores = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))

# 构建特征
def build_features(df, model, method='mean'):
    vectors = []
    for tokens in df['tokens']:
        if method == 'mean':
            vectors.append(doc_vector_mean(tokens, model))
        elif method == 'tfidf':
            vectors.append(doc_vector_tfidf(tokens, model, idf_scores))
    return np.vstack(vectors)

# 构建所有模型和方法组合特征
results = []
classifiers = {
    'SVM': LinearSVC(),
    'LogReg': LogisticRegression(max_iter=1000),
    'RF': RandomForestClassifier(n_estimators=200, random_state=42)
}

for dim, model in EMBEDDINGS.items():
    for method in ['mean', 'tfidf']:
        print(f"Processing features: {dim}d-{method}")
        X_train = build_features(train_df, model, method)
        y_train = train_df['label']
        X_test = build_features(test_df, model, method)
        y_test = test_df['label']

        for clf_name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            results.append({'dim': dim, 'method': method, 'clf': clf_name, 'acc': acc, 'f1': f1})
            print(f"{dim}d-{method}-{clf_name} | Acc: {acc:.4f} | F1: {f1:.4f}")

# 汇总结果
results_df = pd.DataFrame(results).sort_values(by='f1', ascending=False)
print(results_df)

# 最佳模型混淆矩阵
best = results_df.iloc[0]
print("Best setting:", best.to_dict())

best_model = EMBEDDINGS[best['dim']]
X_train_b = build_features(train_df, best_model, best['method'])
X_test_b = build_features(test_df, best_model, best['method'])
y_train_b = train_df['label']
y_test_b = test_df['label']

clf = classifiers[best['clf']]
clf.fit(X_train_b, y_train_b)
y_pred_b = clf.predict(X_test_b)

cm = confusion_matrix(y_test_b, y_pred_b, labels=labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

