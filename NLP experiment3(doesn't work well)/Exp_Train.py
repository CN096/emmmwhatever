import torch
import torch.nn as nn
import time
import json
import os

from tqdm import tqdm
from torch.utils.data import  DataLoader
from Exp_DataSet import Corpus,CorpusBERT
from torch.optim.lr_scheduler import StepLR
from Exp_Model import BiLSTM_model, Transformer_model,BertClassifier


def train():
    '''
    进行训练
    '''
    max_valid_acc = 0
    
    for epoch in range(num_epochs):
        model.train()

        total_loss = []
        total_true = []

        tqdm_iterator = tqdm(data_loader_train, dynamic_ncols=True, desc=f'Epoch {epoch + 1}/{num_epochs}')

        for data in tqdm_iterator:
            # 选取对应批次数据的输入和标签
            batch_x, batch_y = data[0].to(device), data[1].to(device)

            # 模型预测
            y_hat = model(batch_x)

            loss = loss_function(y_hat, batch_y)

            optimizer.zero_grad()   # 梯度清零
            loss.backward()         # 计算梯度
            optimizer.step()        # 更新参数

            y_hat = torch.tensor([torch.argmax(_) for _ in y_hat]).to(device)
            
            total_true.append(torch.sum(y_hat == batch_y).item())
            total_loss.append(loss.item())

            tqdm_iterator.set_postfix(loss=sum(total_loss) / len(total_loss),
                                      acc=sum(total_true) / (batch_size * len(total_true)))
        
        tqdm_iterator.close()

        train_loss = sum(total_loss) / len(total_loss)
        train_acc = sum(total_true) / (batch_size * len(total_true))

        valid_acc = valid()

        if valid_acc > max_valid_acc:
            max_valid_acc=valid_acc
            torch.save(model, os.path.join(output_folder, "model3.ckpt"))

        print(f"epoch: {epoch}, train loss: {train_loss:.4f}, train accuracy: {train_acc*100:.2f}%, valid accuracy: {valid_acc*100:.2f}%")


def valid():
    '''
    进行验证，返回模型在验证集上的 accuracy
    '''
    total_true = []

    model.eval()
    with torch.no_grad():
        for data in tqdm(data_loader_valid, dynamic_ncols=True):
            batch_x, batch_y = data[0].to(device), data[1].to(device)

            y_hat = model(batch_x)
            # 取分类概率最大的类别作为预测的类别
            y_hat = torch.tensor([torch.argmax(_) for _ in y_hat]).to(device)

            total_true.append(torch.sum(y_hat == batch_y).item())

        return sum(total_true) / (batch_size * len(total_true))


def predict():
    '''
    读取训练好的模型对测试集进行预测，并生成结果文件
    '''
    test_ids = [] 
    test_pred = []

    model = torch.load(os.path.join(output_folder, "model.ckpt")).to(device)
    model.eval()
    with torch.no_grad():
        for data in tqdm(data_loader_test, dynamic_ncols=True): 
            batch_x, batch_y = data[0].to(device), data[1]

            y_hat = model(batch_x)
            y_hat = torch.tensor([torch.argmax(_) for _ in y_hat])

            test_ids += batch_y.tolist()
            test_pred += y_hat.tolist()

    # 写入文件
    with open(os.path.join(output_folder, "predict.json"), "w") as f:
        for idx, label_idx in enumerate(test_pred):
            one_data = {}
            one_data["id"] = test_ids[idx]
            one_data["pred_label_desc"] = dataset.dictionary.idx2label[label_idx][1]
            json_data = json.dumps(one_data)    # 将字典转为json格式的字符串
            f.write(json_data + "\n")
            

if __name__ == '__main__':
    dataset_folder = 'D:\\Users\\CN096\\NLP\\tnews-2024\\tnews_public\\'
    output_folder = 'D:\\Users\\CN096\\NLP\\tnews-2024\\model'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    #-----------------------------------------------------begin-----------------------------------------------------#
    # 以下为超参数，可根据需要修改
    embedding_dim = 300     # 每个词向量的维度
    max_token_per_sent = 30 # 每个句子预设的最大 token 数
    batch_size = 128
    num_epochs = 10
    lr = 2e-4   

    #------------------------------------------------------end------------------------------------------------------#

    dataset = Corpus(dataset_folder, max_token_per_sent)
    corpus = CorpusBERT(dataset_folder, max_token_per_sent)
    embedding_weight = torch.tensor(dataset.embedding_weight, dtype=torch.float32)
    vocab_size = len(dataset.dictionary.tkn2word)+2

    print("embedding_weight.shape:", embedding_weight.shape)
    print("vocab_size:", vocab_size)
    print("embedding_dim:", embedding_dim)


    data_loader_train = DataLoader(dataset=dataset.train, batch_size=batch_size, shuffle=True)
    data_loader_valid = DataLoader(dataset=dataset.valid, batch_size=batch_size, shuffle=False)
    data_loader_test = DataLoader(dataset=dataset.test, batch_size=batch_size, shuffle=False)

    #-----------------------------------------------------begin-----------------------------------------------------#
    # 可修改选择的模型以及传入的参数
    #model = BiLSTM_model(vocab_size=vocab_size, ntoken=max_token_per_sent, d_emb=embedding_dim,embedding_weight=embedding_weight).to(device)
    model = Transformer_model(vocab_size=vocab_size, ntoken=max_token_per_sent, d_emb=embedding_dim,embedding_weight=embedding_weight).to(device) 
    #model = BertClassifier().to(device)                           
    #------------------------------------------------------end------------------------------------------------------#
    
    # 设置损失函数
    loss_function = nn.CrossEntropyLoss()
    # 设置优化器                                       
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)  
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)
    
    scheduler = StepLR(optimizer, step_size=4, gamma=0.5)  # 每3个epoch学习率减半

    # 进行训练
    train()

    # 对测试集进行预测
    predict()
