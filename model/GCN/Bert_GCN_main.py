# -*- coding: UTF-8 -*-
import sys

PATH1 = '/root/MLabelCls'
if PATH1 not in sys.path:
    sys.path.insert(0, PATH1)

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
import pandas as pd
from sklearn import metrics

from GCN.Graph_init import *
from GCN.GCN_dataLoader import *
import math
from GCN.util import *
from GCN.Bert_GCN import Bert_GCN

# *----------- For Neural Network -----------------*
def train(model, train_X, loss_func, optimizer , device = 'cpu' , opt = Bert_Config() , target_aux = None):
    model.train()
    total_loss = 0
    for i, batch in enumerate(train_X):
        if opt.extra_loss:
            x, mask, y , y_aux = batch
            # y_aux = y_aux.squeeze(1).to(device) # 原单标签多分类任务 CrossEntropy
        else:
            x, mask, y = batch
        optimizer.zero_grad()
        input_x  = x.to(device)
        mask = mask.to(device)

        y = y.to(device)
        output = model(input_x , mask)
        if opt.extra_loss:
            # NOTICE: 此处 top_k = 2 指只用前俩标签作cross entropy loss
            loss_func_aux = AuxLoss(top_k=2, device = device , alpha_Belance=opt.alpha_Balenced , gamma_Focal=opt.gamma_Focal ) # nn.CrossEntropyLoss() # AuxLos
            # s()  weight=torch.tensor([0.8 , 0.2])
            output = model(input_x , mask)
            loss = loss_func( torch.sigmoid( torch.log_softmax(output , dim=-1) ) , y) # torch.sigmoid(output)
            loss_aux = loss_func_aux(output, y_aux , target_aux) if opt.aux_littleLabels else loss_func_aux(output, y_aux)
            loss = opt.lambda_factor * loss + (1-opt.lambda_factor) * loss_aux
        else:
            output = torch.sigmoid(output)  # torch.softmax(output , dim=-1)  # for BCE_loss
            loss = loss_func(output, y)
        # backward and optimize
        loss.backward()
        optimizer.step()

        # 自适应版，设置 alpha 应该被mask 并额外关注的 label, 提高 label-F1
        # batch_size = output.shape[0]
        if i == 0:
            y_pred = torch.argmax( output , dim = -1 )
            C = confusion_matrix( y_aux[:,0].cpu().numpy() , y_pred.cpu().numpy(),labels= np.arange(opt.label_num) )
            # a = torch.randn((5,7))
            # _, indice = torch.topk(a , k = opt.top_k , dim=-1 )
            # # _, indice = torch.sort( output , dim=-1 , descending = True)
            # macro = [ [0 , 0 , 0] for _ in range(opt.label_num) ] # predict TP , FP , FN
            # for i in range(batch_size): pass
        else:
            y_pred = torch.argmax(output, dim=-1)
            C += confusion_matrix( y_aux[:,0].cpu().numpy() , y_pred.cpu().numpy(),labels= np.arange(opt.label_num))

        total_loss += loss.item() * output.shape[0]
        # if (i + 1) % 300 == 0:
        #     print("Step [{}/{}] Train Loss: {:.4f}".format(i + 1, len(train_X), loss.item()))
    print("train loss: " , total_loss / len(train_X.dataset))
    print('acc:' , C.trace() / C.sum() )
    return total_loss / len(train_X.dataset)


def predict(model,test_X, device = 'cpu' ,opt = Bert_Config() ):
    '''
    :return:
        y_pred: [batch , predicted multi - labels under threshold (Maxi length is Top_k) ]
        y_true: ground-truth one hot
    '''
    model.eval()
    with torch.no_grad():
        for i , batch in enumerate(test_X):
            if opt.extra_loss:
                x, mask, y , y_aux = batch
                # y_aux = y_aux.squeeze(1).to(device)
            else:
                x, mask, y = batch

            input_x  = x.to(device)
            mask = mask.to(device)
            output = model( input_x , mask)
            output = torch.sigmoid(output)

            pred_prob = output.detach()  # .to('cpu').numpy()
            v, idx = torch.topk(pred_prob, k=opt.top_k)  #
            # true_Threshold , 任务改变，多选题变为固定选3项的多选（新高考化学套路）
            # idx = idx.masked_fill(v < opt.true_Threshold, -100)

            if i == 0:
                y_pred = idx.to('cpu').numpy()
                y_true = y
            else:
                y_pred = np.concatenate((y_pred , idx.to('cpu').numpy() ) )
                y_true = np.concatenate((y_true,y))
    return y_pred , y_true



def main(File_Paths = {} , opt = None , MODE = 0):
    if opt is not None:
        print('already set opt config!')
    else:
        # step 1: 数据集 get input
        opt = Bert_Config()
        extra = {'If_GCN':True, 'User_comments':True, 'Only_Comments':False, 'epoch': 9, 'With_abstract':True, 'use_cuda':True, 'learning_rate': 5e-6 , 'true_Threshold': 0.0 ,
                 'extra_loss': True, 'lambda_factor':0, 'GCN_emb_size': 768, 'gamma_Threshold': 0.1 ,
                 'maxi_len_with_label':128 , 'random_seed': 256,  'gamma_Focal':2,
                 'alpha_Balenced':0.25, 'aux_littleLabels':False, 'GCN_layer':1} # 额外参数传递

        opt.parse(extra)

    # print(opt.gamma_Focal)
    CURRENT_SPARE_GPU , free_rate = Spare_gpu((0,1,2))  # 2
    print("gpu_id {} , free_rate:{} ".format( CURRENT_SPARE_GPU , free_rate) )
    if free_rate <= 0.5:
        print('gpu 内存或许不足！')
        assert False
    # 数据集划分
    df = pd.read_csv( File_Paths['df'] )
    df_kw = pd.read_csv( File_Paths['df_kw'] )

    if 'index' in df.columns:
        Indexs = list(df['index'])
        aux_file_path = '/root/MLabelCls/data/Hashtag_emoji/keywords_3_10.csv'
        aux_keyword = pd.read_csv(aux_file_path)
        df_kw = aux_keyword.iloc[Indexs]

    def labels_process( df, df_kw ):
        # 遍历
        record_idx = []
        for i in range(len(df)):
            raw_label = df['raw_labels'].iloc[i]
            if len( eval(raw_label) ) == 3:
                record_idx.append(i)
        return df.iloc[record_idx] , df_kw.iloc[record_idx]

    df , df_kw = labels_process(df , df_kw)
    # df = pd.read_csv("/root/MLabelCls/data/Hashtag_emoji/HashtagAndLabels_05.csv")
    # df_kw = pd.read_csv("/root/MLabelCls/data/Hashtag_emoji/keywords_test.csv")

    # # shrink
    # df = df.iloc[0:500]
    # df_kw = df_kw.iloc[0:500]

    df_train, df_valid , df_test, df_kw_train, df_kw_valid , df_kw_test = split_train_test(df, df_kw , 0.1, 0.05)
    print(len(df_train) , len(df_valid) , len(df_test) )
    setup_seed(opt.random_seed)
    # raw_train_labels = list( df_train['one_hot'].apply(oneHot2raw) ) # oneHot2raw
    raw_train_labels = list(df_train['raw_labels'].apply(lambda x: eval(x)))
    adj = get_adjMat(raw_train_labels , opt) # TODO: get_adj, Gamma_threshold  # 调参？

    # save_path = "/root/MLabelCls/GCN/A_rewight_gamma_1p_2.npy"
    # adj = np.load(save_path) # 行和为1 , （实际上，多数网络是用的列和为1）
    # # 由于 re-weighted adj , col sum = 1 , \therefore,  D = I ; 假装已经搞好了

    adj = torch.tensor(adj, dtype = torch.float32)

    label_num = adj.shape[0]
    # label_emb = get_label_embedding(label_num , emb_size=opt.GCN_emb_size) # Graph中的 X 矩阵, (衡量节点特征)
    # to test
    # label_emb = nn.Parameter( label_emb.detach() )


    # step 2: data loader 相关
    tokenizer = BertTokenizer.from_pretrained(File_Paths['tokenizer']) # "/root/Bert_preTrain"
    # tokenizer.encode_plus('崔子悦')
    train_set = get_Bert_dataset(df_train , tokenizer ,df_kw_train , opt = opt) if opt.User_comments else get_Bert_dataset(df_train , tokenizer, opt = opt)
    train_dl = data.DataLoader(train_set, batch_size=opt.batch_size, drop_last=False, shuffle=True)

    valid_set = get_Bert_dataset(df_valid, tokenizer, df_kw_valid, opt=opt) if opt.User_comments else get_Bert_dataset(df_valid, tokenizer, opt=opt)
    valid_dl = data.DataLoader(valid_set, batch_size=opt.batch_size, drop_last=False, shuffle=True)

    test_set = get_Bert_dataset(df_test, tokenizer, df_kw_test, opt=opt) if opt.User_comments else get_Bert_dataset(df_test, tokenizer, opt=opt)
    test_dl = data.DataLoader(test_set, batch_size=opt.batch_size, drop_last=False, shuffle=False)

    print('data loader', len(train_dl) , len(valid_dl), len(test_dl)  )
    # step 3: 模型、训练所需
    if torch.cuda.is_available() and opt.use_cuda:
        if MODE == 0: # for SSH
            device = torch.device("cuda:{}".format(CURRENT_SPARE_GPU) if torch.cuda.is_available() else "cpu")
        else: # for colab
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )
    else:
        device = 'cpu'
    print('final device: ' , device)
    bert_model = BertModel.from_pretrained(File_Paths['bert']) # https://github.com/ymcui/Chinese-BERT-wwm

    # ”简单地“引入 labels的语义信息, 不增反降
    label_emb = get_label_embedding(label_num , emb_size=opt.GCN_emb_size , method = 'random') # Graph中的 X 矩阵, (衡量节点特征)
    # opt.GCN_emb_size = 300
    # label_emb = get_label_embedding(label_num , emb_size=opt.GCN_emb_size , method = 'word_vc')
    # label_emb = get_label_embedding(label_num , emb_size=opt.GCN_emb_size , method = 'bert_encoder', model = bert_model , tokenizer = tokenizer, device = 'cpu' ) # Graph中的 X 矩阵, (衡量节点特征)

    model = Bert_GCN(bert_model , opt , label_embedding=label_emb , A=adj)
    # TODO 心血来潮
    # reason_model = Bert_GCN(bert_model, opt_reasoner, A=torch.randn( (opt_reasoner.label_num, opt_reasoner.label_num)  ))  # Reasoner(opt_reasoner) # 直白 GCN
    # Model_PATH = File_Paths['use_brother'] # 纯粹心血来潮试试
    # model.load_state_dict(torch.load(Model_PATH))
    # reason_model.to(device)

    print('get here!', device)
    model.to(device)
    print(model)
    print('model to {} ... '.format(device))
    optimizer = torch.optim.Adam( model.parameters(), lr = opt.learning_rate , weight_decay = opt.learning_rate_decay )
    loss_func = nn.BCELoss() # loss(sigmoid(x) , target)
    # scheduler = CosineAnnealingLR(optimizer, opt.epoch)

    print("模型配置完成! ")
    # step 4: 炼！
    print('start ...')
    start = time.time()
    last_instanceF1 = None
    target_aux = None
    best_y_pred = None

    for epoch in range(opt.epoch):
        train(model, train_X=train_dl, optimizer=optimizer , loss_func=loss_func , device = device , opt = opt , target_aux = target_aux)

        # y_pred , y_true = predict(model , test_X=valid_dl , device = device, opt=opt)
        # y_pre = idx2binary(y_pred , opt)
        # print('*' * 20)
        # print(y_pre[0] , y_true[0])
        # print('*' * 20)
        # metrics = evaluate(y=y_true, y_pre=y_pre)
        # print('Valid metrics: ' , metrics)

        y_pred , y_true = predict(model , test_X=test_dl , device = device, opt=opt)
        y_pre = idx2binary(y_pred , opt)
        print('*' * 20)
        print(y_pre[0] , y_true[0])
        print('*' * 20)
        metrics = evaluate(y=y_true, y_pre=y_pre)
        print( metrics )
        if (last_instanceF1 is None) or (last_instanceF1 < metrics['instance_f1']):
            last_instanceF1 = metrics['instance_f1']
            if opt.record_model and 'model_path' in File_Paths:
                Model_PATH = File_Paths['model_path']
                torch.save(model.state_dict(), Model_PATH)
                print('model saved here... ')
            # 可视化研究用 y_pred
            best_y_pred = y_pred
        elif epoch > 6:
            break

        if opt.aux_littleLabels:
            _ , indices = torch.topk( -torch.tensor( metrics['all_class'] ) , k = opt.label_num // 2 )
            target_aux = torch.zeros( opt.label_num , dtype=torch.long )
            target_aux[indices] = 1 # balenced labels
            # target_aux = target_aux.to(device)

        print("epoch: {} , 总时间: {} ".format( epoch , time.time() - start)  )
        print("# *------ new epoch ------------* # ")
        # scheduler.step()

    return best_y_pred , y_true

def data_augmentation_analyze():
    opt = Bert_Config()
    extra = {'If_GCN': True, 'User_comments':False , 'Only_Comments': False, 'epoch': 9, 'With_abstract':False ,
             'use_cuda': True, 'learning_rate': 5e-6, 'true_Threshold': 0.0,
             'extra_loss': True, 'lambda_factor': 0, 'GCN_emb_size': 768, 'gamma_Threshold': 0.1,
             'maxi_len_with_label': 128, 'random_seed': 256, 'gamma_Focal': 0,
             'alpha_Balenced': 0, 'aux_littleLabels': False, 'GCN_layer': 1}  # 额外参数传递

    opt.parse(extra)

    File_Paths = {'df': "/root/MLabelCls/data/Hashtag_emoji/HLA_39.csv",
                  # "/root/MLabelCls/data/Hashtag_emoji/HashtagAndLabels_00.csv",
                  'df_kw': '/root/MLabelCls/data/Hashtag_emoji/keywords_3_10.csv',
                  'tokenizer': "/root/Bert_preTrain",
                  'bert': "/root/Bert_preTrain"}
    Need_train = False
    # save path
    save_path_1 = '/Data_storage/outcome/hash.npy'
    save_path_2 = '/Data_storage/outcome/abst.npy'
    save_path_3 = '/Data_storage/outcome/uc.npy'

    if Need_train:
        y_hash, y_true = main(File_Paths , opt)

        opt.With_abstract = True
        y_abst, _ = main(File_Paths, opt)

        opt.With_abstract = False
        opt.User_comments = True
        y_uc, _ = main(File_Paths , opt)


        np.save(save_path_1 , y_hash)
        np.save(save_path_2 , y_abst)
        np.save(save_path_3 , y_uc)

    y_hash , y_abst , y_uc = np.load(save_path_1) , np.load(save_path_2) , np.load(save_path_3)

    df = pd.read_csv( File_Paths['df'] )
    df_kw = pd.read_csv( File_Paths['df_kw'] )

    if 'index' in df.columns:
        Indexs = list(df['index'])
        aux_file_path = '/root/MLabelCls/data/Hashtag_emoji/keywords_3_10.csv'
        aux_keyword = pd.read_csv(aux_file_path)
        df_kw = aux_keyword.iloc[Indexs]

    def labels_process( df, df_kw ):
        # 遍历
        record_idx = []
        for i in range(len(df)):
            raw_label = df['raw_labels'].iloc[i]
            if len( eval(raw_label) ) == 3:
                record_idx.append(i)
        return df.iloc[record_idx] , df_kw.iloc[record_idx]

    df , df_kw = labels_process(df , df_kw)
    df_train, df_valid, df_test, df_kw_train, df_kw_valid, df_kw_test = split_train_test(df, df_kw, 0.1, 0.05)
    # assert len(df_test) == len(y_true)
    y_true = list(df_test['raw_labels'].apply(lambda x: eval(x)))
    y_true = idx2binary(np.array(y_true) , opt)
    # print(y_true.shape , len(df_test))

    # 孙俪新剧《理想之城》预告曝光，看完预告想给孙俪演技点赞！
    a = np.array( list(df_test['hashtag'].apply( lambda x: len(x))) )


    def better( y_1 , y_2, ground_truth):
        # y_1 相比 y_2 强的地方
        record_idx = []
        for i in range(len(y_1)):
            if ground_truth[i][y_1[i]].sum() > ground_truth[i][y_2[i]].sum() :
                record_idx.append(i)
        return record_idx

    # def calc_num(y , ground_truth):
    #     record_idx = []
    #     for i in range(len(y_1)):
    #         if ground_truth[i][y_1[i]].sum() > ground_truth[i][y_2[i]].sum() :
    #             record_idx.append(i)
    #     pass
    print(len(df_test))
    missing_idx = []
    for i in range(len(df_test)):
        if y_true[i][ y_hash[i] ].sum() < 3:
            missing_idx.append(i)
    missing = len(missing_idx)
    missing_idx = set(missing_idx)
    print('有提升空间的数目: ' , missing)

    record_idx_1 = better(y_abst , y_hash , y_true)
    print('abstract： ',len(set(record_idx_1) & missing_idx))
    hash_len = list( df_test['hashtag'].iloc[record_idx_1].apply(lambda x: len(x))  )
    print('1、 增强的平均长度:  ' , sum(hash_len) / len(hash_len) )

    record_idx_2 = better(y_uc, y_hash, y_true)
    print('comment： ',len(set(record_idx_2) & missing_idx))
    hash_len = list( df_test['hashtag'].iloc[record_idx_2].apply(lambda x: len(x))  )
    print('2、 增强的平均长度:  ' , sum(hash_len) / len(hash_len) )

    print('整体占比: ' ,  len(set(record_idx_1) & missing_idx) /missing, len(set(record_idx_2) & missing_idx)/ missing ) #  len(df_test)

    # 验证是否对短文本更有效, 结果是，无所谓
    idx = set( np.where(a<=6)[0]) # 6字以下
    record_idx_1 = set(record_idx_1)
    record_idx_2 = set(record_idx_2)

    print( len( record_idx_1 & idx )  ,len( record_idx_1 & idx ) / len(idx) ) # 5字以下 12 / 51 ; 20 / 86
    print(len( record_idx_2 & idx ) ,len( record_idx_2 & idx )  / len(idx)) # 5字以下 17 / 51 ; 26 / 86
    print( '短文本数' , len(idx))

    positive = {'[赞]': 5, '[给你小心心]': 17, '[嘻嘻]': 18, '[心]': 0, '[给力]': 13, '[中国赞]': 14, '[憧憬]': 7, '[威武]': 23,
                '[笑cry]': 8, '[酸]': 20, }
    negative = {'[泪]': 4, '[跪了]': 16, '[伤心]': 12, '[怒]': 6, '[拜拜]': 19, '[蜡烛]': 10}
    neural = {'[doge]': 2, '[允悲]': 3, '[并不简单]': 21, '[思考]': 22, '[吃惊]': 15, '[费解]': 1, '[微笑]': 11, '[吃瓜]': 9}
    positive, negative, neural = set(positive.values()), set(negative.values()), set(neural.values())

    # 验证是否对复杂情感更有效?
    print('是否对复杂情感更有效?')
    idx = set()
    for i in range(len(df_test)):
        emoji_s = eval(df_test['raw_labels'].iloc[i])
        tmp_set = set(emoji_s[0:2])
        if int(len(positive & tmp_set) and len(negative & tmp_set)):
            idx.add(i)

    print( len( record_idx_1 & idx ) , len( record_idx_1 & idx )/len(idx) )
    print( len( record_idx_2 & idx ) , len( record_idx_2 & idx ) / len(idx))
    print('总混淆数', len(idx) )


    # 卡方检验
    from scipy import stats
    # 列联表
    # stats.chisquare(f_obs=observed,  # Array of obversed counts
    #                 f_exp=expected)  # Array of expected counts
    a, b, c, d, n = [26,185,60,504,775]
    Chi_square = n * (a * d - b * c) ** 2 / ((a + b) * (c + d) * (a + c) * (b + d))

    return



if __name__ == '__main__':
    print('okk, 5_24 get busy living, or get busy dying')
    import sys , os
    print(sys.path )
    print(os.path)
    print(torch.__version__)

    MODE = 0
    if MODE == 0: # ssh
        File_Paths = {'df':  "/root/MLabelCls/data/Hashtag_emoji/HLA_39.csv", # "/root/MLabelCls/data/Hashtag_emoji/HashtagAndLabels_00.csv",
                      'df_kw': '/root/MLabelCls/data/Hashtag_emoji/keywords_3_10.csv',
                      'tokenizer': "/root/Bert_preTrain",
                      'bert' : "/root/Bert_preTrain" }
    elif MODE == 1: # 本地调试模式
        File_Paths = {'df': "F:\\specific_python\\FinalGame\\data\\HashtagAndLabels_00.csv",
                      'df_kw': "F:\\specific_python\\FinalGame\\data\\keywords_test.csv",
                      'tokenizer': "F:\\SentimentCls\\Bert预训练",
                      'bert': "F:\\SentimentCls\\Bert预训练" }
    elif MODE==2: # colab
        File_Paths = {'df': "/content/drive/MyDrive/diploma_project/data/HashtagAndLabels_00.csv",
                      'df_kw': "/content/drive/MyDrive/diploma_project/data/keywords_test.csv",
                      'tokenizer': "/content/drive/MyDrive/BERT_preTrain",
                      'bert': "/content/drive/MyDrive/BERT_preTrain"
                      }
    # main(File_Paths)

    data_augmentation_analyze()