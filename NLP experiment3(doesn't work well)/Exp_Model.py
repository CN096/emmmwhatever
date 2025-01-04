import torch.nn as nn
import torch as torch
import math
from transformers import BertModel, BertTokenizer

class BertClassifier(nn.Module):
    def __init__(self, num_classes=15, dropout=0.3):
        super(BertClassifier, self).__init__()
        
        # 加载预训练的 BERT-base 模型
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # 定义分类器层
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)  # 输出维度与分类数目一致

    def forward(self, input_ids, attention_mask):
        # 获取 BERT 的最后一层输出和池化输出
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output  # [CLS] token 的输出特征
        
        # Dropout + 全连接分类
        x = self.dropout(cls_output)
        x = self.fc(x)
        return x
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.2, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Transformer_model(nn.Module):
    def __init__(self, vocab_size, ntoken, d_emb=512, d_hid=2048, nhead=4, nlayers=6, dropout=0.3, embedding_weight=None, num_classes=15):
        super(Transformer_model, self).__init__()
        # 将"预训练的词向量"整理成 token->embedding 的二维映射矩阵 emdedding_weight 的形式，初始化 _weight
        # 当 emdedding_weight == None 时，表示随机初始化
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_emb, _weight=embedding_weight)

        self.pos_encoder = PositionalEncoding(d_model=d_emb, max_len=ntoken)
        self.encode_layer = nn.TransformerEncoderLayer(d_model=d_emb, nhead=nhead, dim_feedforward=d_hid,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encode_layer, num_layers=nlayers)
        #-----------------------------------------------------begin-----------------------------------------------------#
        # 添加自注意力机制作为隐藏层数据处理
        self.attention_weights = nn.Linear(d_emb, 1)  # 计算注意力权重
        self.dropout = nn.Dropout(dropout)  # Dropout层

        # 使用两层分类器进行特征处理
        self.fc1 = nn.Linear(d_emb, d_emb // 2)  # 降维
        self.fc2 = nn.Linear(d_emb // 2, num_classes)  # 输出分类
        self.activation = nn.ReLU()  # 激活函数
        #------------------------------------------------------end------------------------------------------------------#


    def forward(self, x):
        x = self.embed(x)     
        x = x.permute(1, 0, 2)          
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
                #-----------------------------------------------------begin-----------------------------------------------------#
        # 使用自注意力机制选择重要特征
        attention_weights = torch.softmax(torch.matmul(x, x.transpose(-2, -1)), dim=-1)  # 自注意力权重
        attended_x = torch.matmul(attention_weights, x)  # 加权后的特征

        # 使用最大池化提取最显著特征
        pooled_x = torch.max(attended_x, dim=1).values  # 最大池化

        # 将池化后的特征送入两层分类器
        x = self.dropout(pooled_x)  # Dropout
        x = self.fc1(x)  # 第一层全连接降维
        x = self.activation(x)  # 激活函数
        x = self.fc2(x)  # 第二层全连接分类
        #------------------------------------------------------end------------------------------------------------------#


        return x
    
    
class BiLSTM_model(nn.Module):
    def __init__(self, vocab_size, ntoken, d_emb=100, d_hid=80, nlayers=1, dropout=0.4, embedding_weight=None):
        super(BiLSTM_model, self).__init__()
        # 初始化嵌入层，使用预训练词向量或随机初始化
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_emb, _weight=embedding_weight)

        # BiLSTM 层
        self.lstm = nn.LSTM(input_size=d_emb, hidden_size=d_hid, num_layers=nlayers, 
                            bidirectional=True, batch_first=True)
        #-----------------------------------------------------begin-----------------------------------------------------#
        # 隐藏层数据的处理和分类器设计
        self.dropout = nn.Dropout(dropout)  # 防止过拟合
        self.hidden_size = d_hid * 2  # 双向 LSTM 的隐藏层大小

        # 分类器设计
        self.fc = nn.Linear(self.hidden_size, 15)  # 将隐藏层输出映射到类别数量
        self.softmax = nn.LogSoftmax(dim=-1)  # 计算类别概率分布（对数形式）
        #------------------------------------------------------end------------------------------------------------------#

    def forward(self, x):
        # 嵌入层
        x = self.embed(x)

        # BiLSTM 层
        lstm_out, _ = self.lstm(x)  # lstm_out 的形状为 (batch_size, seq_len, hidden_size * 2)

        #-----------------------------------------------------begin-----------------------------------------------------#
        # 对 BiLSTM 隐藏层输出进行处理和选择
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        lstm_out = self.layer_norm(lstm_out)
        # 采用混合池化方法
        max_pool = torch.max(lstm_out, dim=1).values
        mean_pool = torch.mean(lstm_out, dim=1)
        x = 0.5 * max_pool + 0.5 * mean_pool  # 平均结合


        # 添加 Dropout 防止过拟合
        self.dropout = nn.Dropout(p=0.5)
        x = self.dropout(x)
        

        # 全连接层映射到类别
        x = self.fc(x)

        # Softmax 输出
        x = self.softmax(x)
        #------------------------------------------------------end------------------------------------------------------#

        return x

