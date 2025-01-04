import os
import json
import numpy as np
import json
import torch
import jieba
from torch.utils.data import TensorDataset
from gensim.models.keyedvectors import KeyedVectors

class Dictionary(object):
    def __init__(self, path):

        self.word2tkn = {"[PAD]": 0}
        self.tkn2word = ["[PAD]"]

        self.label2idx = {}
        self.idx2label = []

        # 获取 label 的 映射
        with open(os.path.join(path, 'labels.json'), 'r', encoding='utf-8') as f:
            for line in f:
                one_data = json.loads(line)
                label, label_desc = one_data['label'], one_data['label_desc']
                self.idx2label.append([label, label_desc])
                self.label2idx[label] = len(self.idx2label) - 1

    def add_word(self, word):
        if word not in self.word2tkn:
            self.tkn2word.append(word)
            self.word2tkn[word] = len(self.tkn2word) - 1
        return self.word2tkn[word]


class Corpus(object):
    '''
    完成对数据集的读取和预处理，处理后得到所有文本数据的对应的 token 表示及相应的标签。
    
    该类适用于任务一、任务二，若要完成任务三，需对整个类进行调整，例如，可直接调用预训练模型提供的 tokenizer 将文本转为对应的 token 序列。
    '''
    def __init__(self, path, max_token_per_sent):
        self.dictionary = Dictionary(path)

        self.max_token_per_sent = max_token_per_sent

        self.train = self.tokenize(os.path.join(path, 'train.json'))
        self.valid = self.tokenize(os.path.join(path, 'dev.json'))
        self.test = self.tokenize(os.path.join(path, 'test.json'), True)

        #-----------------------------------------------------begin-----------------------------------------------------#
        # 若要采用预训练的 embedding, 需处理得到 token->embedding 的映射矩阵 embedding_weight。矩阵的格式参考 nn.Embedding() 中的参数 _weight
        # 注意，需考虑 [PAD] 和 [UNK] 两个特殊词向量的设置
                # 读取预训练的字向量
        # 加载预训练的 word2vec 模型
        embedding_path = "D:\\Users\\CN096\\NLP\\tnews-2024\\sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5.bz2"
        word_vectors = KeyedVectors.load_word2vec_format(embedding_path, binary=False)

        embedding_dim = word_vectors.vector_size

        # 构造 token->embedding 的映射矩阵
        vocab_size = len(self.dictionary.word2tkn) + 2  # 考虑 [PAD] 和 [UNK]
        embedding_weight = np.random.uniform(-0.1, 0.1, (vocab_size, embedding_dim)).astype(np.float32)

        #pad_idx = self.dictionary.add_word('[PAD]')
        #unk_idx = self.dictionary.add_word('[UNK]')
        # 添加特殊词
        pad_idx = self.dictionary.word2tkn['[PAD]'] = 0
        unk_idx = self.dictionary.word2tkn['[UNK]'] = 1

        # 同步更新 tkn2word
        self.dictionary.tkn2word[pad_idx] = '[PAD]'
        self.dictionary.tkn2word[unk_idx] = '[UNK]'

        embedding_weight[pad_idx] = np.zeros(embedding_dim, dtype=np.float32)
        embedding_weight[unk_idx] = np.mean([word_vectors[word] for word in word_vectors.key_to_index], axis=0)

        # 补充缺失的 tkn2word 映射
        for word, token_id in self.dictionary.word2tkn.items():
            if token_id not in self.dictionary.tkn2word:
                self.dictionary.tkn2word[token_id] = word
        # 输出 vocab_size
        

        # 填充预训练向量
        for word, idx in self.dictionary.word2tkn.items():
            if word in word_vectors.key_to_index:
                embedding_weight[idx] = word_vectors[word]

        self.embedding_weight = embedding_weight


        #------------------------------------------------------end------------------------------------------------------#

    def pad(self, origin_token_seq):
        '''
        padding: 将原始的 token 序列补 0 至预设的最大长度 self.max_token_per_sent
        '''
        if len(origin_token_seq) > self.max_token_per_sent:
            return origin_token_seq[:self.max_token_per_sent]
        else:
            return origin_token_seq + [0 for _ in range(self.max_token_per_sent-len(origin_token_seq))]

    def tokenize(self, path, test_mode=False):
        '''
        处理指定的数据集分割，处理后每条数据中的 sentence 都将转化成对应的 token 序列。
        '''
        idss = []
        labels = []
        with open(path, 'r', encoding='utf8') as f:
            for line in f:
                one_data = json.loads(line)  # 读取一条数据
                sent = one_data['sentence']
                #-----------------------------------------------------begin-----------------------------------------------------#
                # 若要采用预训练的 embedding, 需在此处对 sent 进行分词
                words = list(jieba.cut(sent))  # 使用 jieba 分词

                #------------------------------------------------------end------------------------------------------------------#
                # 向词典中添加词
                for word in words:
                    self.dictionary.add_word(word)

                ids = []
                for word in words:
                    ids.append(self.dictionary.word2tkn[word])
                idss.append(self.pad(ids))
                
                # 测试集无标签，在 label 中存测试数据的 id，便于最终预测文件的打印
                if test_mode:
                    label = json.loads(line)['id']      
                    labels.append(label)
                else:
                    label = json.loads(line)['label']
                    labels.append(self.dictionary.label2idx[label])

            idss = torch.tensor(np.array(idss))
            labels = torch.tensor(np.array(labels)).long()
            print(f"Before padding: {len(ids)}, After padding: {len(self.pad(ids))}")

        return TensorDataset(idss, labels)
    

from transformers import BertTokenizer

class CorpusBERT(object):
    '''
    完成对数据集的读取和预处理，将短文本转化为 BERT 的输入格式。
    包括 input_ids, attention_mask 和 label。
    '''
    def __init__(self, path, max_token_per_sent):
        self.max_token_per_sent = max_token_per_sent

        # 初始化 BERT 分词器
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # 加载数据集
        self.train = self.tokenize(os.path.join(path, 'train.json'))
        self.valid = self.tokenize(os.path.join(path, 'dev.json'))
        self.test = self.tokenize(os.path.join(path, 'test.json'), test_mode=True)

    def pad(self, input_ids):
        '''
        确保 token 序列符合最大长度要求，超过截断，不足补 [PAD]。
        '''
        if len(input_ids) > self.max_token_per_sent:
            return input_ids[:self.max_token_per_sent]
        else:
            return input_ids + [0] * (self.max_token_per_sent - len(input_ids))

    def tokenize(self, path, test_mode=False):
        '''
        处理指定数据集，将文本转为 BERT 的输入格式：input_ids 和 attention_mask。
        '''
        input_ids_list = []
        attention_mask_list = []
        labels = []

        with open(path, 'r', encoding='utf8') as f:
            data = json.load(f)  # 加载 JSON 文件

        for one_data in data:
            sentence = one_data['sentence']
            label = one_data['label'] if not test_mode else one_data['id']

            # 使用 BERT 分词器对句子进行编码
            encoded = self.tokenizer(
                sentence,
                max_length=self.max_token_per_sent,
                padding='max_length',
                truncation=True,
                return_tensors="pt"  # 输出 PyTorch 张量
            )
            input_ids_list.append(encoded['input_ids'].squeeze(0))
            attention_mask_list.append(encoded['attention_mask'].squeeze(0))

            # 测试集无标签，记录 ID 以便预测时使用
            if test_mode:
                labels.append(label)
            else:
                labels.append(int(label))  # 假设标签是整数

        input_ids = torch.stack(input_ids_list)
        attention_masks = torch.stack(attention_mask_list)
        labels = torch.tensor(labels).long() if not test_mode else torch.tensor(labels)

        return TensorDataset(input_ids, attention_masks, labels)

