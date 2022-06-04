# -*- coding: UTF-8 -*-
import sys
PATH1 = '/root/MLabelCls'
if PATH1 not  in sys.path:
    sys.path.insert(0,PATH1)

import numpy as np
import torch
import pandas as pd
from reader_comments.util import *

from GCN.Bert_GCN import main
from GCN.util import Bert_Config


import os
import random


# *-----------   一. ALL IN 大法  --------------*
def AllIn_keywords():

    File_Paths = {'df': "/root/MLabelCls/data/Hashtag_emoji/Reader_Comments.csv",
                  'df_kw': '/root/MLabelCls/data/Hashtag_emoji/keywords_CICD.csv',
                  'tokenizer': "/root/Bert_preTrain",
                  'bert': "/root/Bert_preTrain"}

    opt = Bert_Config()
    extra = {'If_GCN': True, 'User_comments': True, 'Only_Comments': False, 'epoch': 9, 'With_abstract': False,
             'use_cuda': True, 'learning_rate': 5e-6, 'true_Threshold': 0.0,
             'extra_loss': True, 'lambda_factor': 0, 'GCN_emb_size': 768, 'gamma_Threshold': 0.1,
             'maxi_len_with_label': 160, 'random_seed': 256, 'gamma_Focal': 0,
             'alpha_Balenced': 0, 'aux_littleLabels': False,
             'GCN_layer': 1}  # 额外参数传递
    opt.parse(extra)

    # 数据量上: keywords length mean 181; median
    main(File_Paths=File_Paths , opt=opt) # 直白默认 opt 参数
    return

# 随便挑几个评论
def random_comments():
    Maxi_len = 160
    # construct df
    save_path = '/root/MLabelCls/data/Hashtag_emoji/RandomComments.csv'
    if save_path not in os.listdir('/root/MLabelCls/data/Hashtag_emoji/'):
        df_origin = pd.read_csv("/root/MLabelCls/data/Hashtag_emoji/Reader_Comments.csv")
        Data_storage = '/Data_storage/CICD_merge/{}.txt'
        record_RandomComments = []
        for i in range(len(df_origin)):
            file_idx = df_origin['abstract_df_idx'].iloc[i]
            Comments = []
            with open( Data_storage.format(file_idx), 'r', encoding='utf-8') as f:
                first_row = True
                for line in f.readlines():
                    if first_row:
                        first_row = False
                    else:
                        str_list = line.strip('\n').split('\t')
                        comment, raw_label = ''.join(str_list[0:-1]), str_list[-1]
                        Comments.append(comment)

            tmp_txt = df_origin['hashtag'].iloc[i]
            length = len(tmp_txt)
            for j in np.random.permutation(len(Comments)):
                if len(Comments[j]) >= (Maxi_len // 2) and j < len(Comments) - 1:
                    continue
                tmp_txt += '[SEP]' + Comments[j]
                length += len(Comments[j])
                if length >= Maxi_len: break

            record_RandomComments.append(tmp_txt)
        df_origin['hashtag'] = pd.Series(record_RandomComments)
        df_origin.to_csv(save_path , index=False, encoding='utf-8')

    File_Paths = {'df': save_path,
                  'df_kw': '/root/MLabelCls/data/Hashtag_emoji/keywords_CICD.csv',
                  'tokenizer': "/root/Bert_preTrain",
                  'bert': "/root/Bert_preTrain",
                  'model_path': '/Data_storage/model/Bert_RandomSample_LM.pt',
                  'use_brother':'/Data_storage/model/Bert_GCN_LM.pt'}

    opt = Bert_Config()
    extra = {'If_GCN': True, 'User_comments': False, 'Only_Comments': False, 'epoch': 9, 'With_abstract': False,
             'use_cuda': True, 'learning_rate': 5e-6, 'true_Threshold': 0.0,
             'extra_loss': True, 'lambda_factor': 0, 'GCN_emb_size': 768, 'gamma_Threshold': 0.1,
             'maxi_len': Maxi_len, 'random_seed': 256, 'gamma_Focal': 0,
             'alpha_Balenced': 0, 'aux_littleLabels': False, 'record_model':True,
             'GCN_layer': 1}  # 额外参数传递
    opt.parse(extra)

    # 数据量上: keywords length mean 181; median
    main(File_Paths=File_Paths , opt=opt) # 直白默认 opt 参数
    return


# 挑选某几种（按一定的优先级）
def random_specificClass_comments():
    # 必选带特定标签的评论 // 仅为验证
    # 现实中，假如已知评论重点情感，则已知答案，无需考试。
    # 此条仅供 baseline 比较
    Maxi_len = 160
    # df = pd.read_csv("/root/MLabelCls/data/Hashtag_emoji/Reader_Comments.csv")
    # for i in range( len(df) ):
    #     original = eval(df['raw_labels'].iloc[i])
    #     ciphertext = eval( df['emoji_s'].iloc[i] ) if isinstance(df['emoji_s'].iloc[i] , str) else df['emoji_s'].iloc[i]
    #     if len(ciphertext) < 3: continue
    #     ciphertext = ciphertext[0:3]
    #     if len( set(original) | set(ciphertext) ) > 3 or len( set(original) ) < len( set(ciphertext) ): # original 有点问题
    #         df.at[i, 'raw_labels'] = str( list(ciphertext) )
    # df.to_csv( "/root/MLabelCls/data/Hashtag_emoji/Reader_Comments.csv" )

    # construct df
    save_path = '/root/MLabelCls/data/Hashtag_emoji/SpecificClass_Comments.csv'
    if save_path not in os.listdir('/root/MLabelCls/data/Hashtag_emoji/'):
        df_origin = pd.read_csv("/root/MLabelCls/data/Hashtag_emoji/Reader_Comments.csv")
        # Data_storage = '/Data_storage/CICD_rawLabel/{}.txt' # NOTICE HERE!!!
        record_RandomComments = []
        for i in range(len(df_origin)):
            file_idx = df_origin['abstract_df_idx'].iloc[i]
            raw_labels = eval( df_origin['raw_labels'].iloc[i] )
            if len(raw_labels) < 3:
                record_RandomComments.append('反正训练时候也要删去...无所谓了')
                continue
            tmp_txt = df_origin['hashtag'].iloc[i]
            Prior = { tuple(raw_labels):0, (raw_labels[0] , raw_labels[1]) :1, (raw_labels[0] ,raw_labels[2]) :2,
                      (raw_labels[0],):3, (raw_labels[1],):4 , (raw_labels[2],):5  }

            Prior_comments = get_CommentFromCIDC_rawLabel(file_idx ,Prior ,Maxi_num=3 )
            length = 0
            for comments in Prior_comments:
                for c in comments:
                    tmp_txt += '[SEP]' +  c
                    length += len(c)
                    if length >= Maxi_len: break
                if length >= Maxi_len: break
            record_RandomComments.append(tmp_txt)
        df_origin['hashtag'] = pd.Series(record_RandomComments)
        df_origin.to_csv(save_path , index=False, encoding='utf-8')

    File_Paths = {'df': save_path,
                  'df_kw': '/root/MLabelCls/data/Hashtag_emoji/keywords_CICD.csv',
                  'tokenizer': "/root/Bert_preTrain",
                  'bert': "/root/Bert_preTrain",
                  'model_path': '/Data_storage/model/Bert_GCN_LM.pt',
                 'use_brother':'/Data_storage/model/Bert_RandomSample_LM.pt',
    }

    opt = Bert_Config()
    extra = {'If_GCN': True, 'User_comments': False, 'Only_Comments': False, 'epoch': 9, 'With_abstract': False,
             'use_cuda': True, 'learning_rate': 5e-6, 'true_Threshold': 0.0,
             'extra_loss': True, 'lambda_factor': 0, 'GCN_emb_size': 768, 'gamma_Threshold': 0.1,
             'maxi_len': Maxi_len, 'random_seed': 256, 'gamma_Focal': 0,
             'alpha_Balenced': 0, 'aux_littleLabels': False,
             'GCN_layer': 1,  'record_model':False}  # 额外参数传递
    opt.parse(extra)

    # TODO: pre-Train GCN
    # 数据量上: keywords length mean 181; median
    main(File_Paths=File_Paths, opt=opt) # 直白默认 opt 参数
    return


# *-----------   二. Merge  法 ---------------*
# 相当于极大似然法
def discreet_Multi_Nominal():
    NEED_MODEL_TRAIN = False # 是否需要重训
    opt = Bert_Config()
    extra = {'label_num':24, 'If_GCN': False, 'User_comments': True, 'Only_Comments': False, 'epoch': 10,
             'use_cuda': True, 'learning_rate': 5e-6, 'true_Threshold': 0.0,
             'extra_loss': True, 'lambda_factor': 0.0, 'GCN_emb_size': 768, 'gamma_Threshold': 0.1,
             'maxi_len': 160, 'random_seed': 256, 'gamma_Focal': 0,
             'alpha_Balenced': 0, 'aux_littleLabels': False,
              'record_model':True , 'NN_cdf':False}  # 额外参数传递
    # 'NN_cdf':True / False 决定了用什么来拟合概率分布
    opt.parse(extra)

    File_Paths = {'df': "/root/MLabelCls/data/Hashtag_emoji/Reader_Comments.csv" ,
                  'df_kw': '/root/MLabelCls/data/Hashtag_emoji/keywords_CICD.csv',
                  'tokenizer': "/root/Bert_preTrain",
                  'bert': "/root/Bert_preTrain",
                  'trainLM_df':'/root/MLabelCls/data/Hashtag_emoji/Comments_CIDC_trainLM.csv',
                  'model_path': '/Data_storage/model/Bert_SingleLM_reweight_3.pt', } # Bert_SingleLM_reweight， 训练一轮*=/三轮

    if NEED_MODEL_TRAIN:
        train_EmojiCls_MLmodel(File_Paths, opt)

    Statistic_Model = 'multi_nominal'
    # Statistic_Model = 'beta'
    predictByEmojiClsLM(File_Paths , opt)
    return

# 神经网络拟合cdf
def discreet_NN():
    # 用neural network 拟合分布
    NEED_MODEL_TRAIN = False # 是否需要重训
    opt = Bert_Config()
    extra = {'label_num':24, 'If_GCN': False, 'User_comments': True, 'Only_Comments': False, 'epoch': 10,
             'use_cuda': True, 'learning_rate': 5e-6, 'true_Threshold': 0.0,
             'extra_loss': True, 'lambda_factor': 0.0, 'GCN_emb_size': 768, 'gamma_Threshold': 0.1,
             'maxi_len': 160, 'random_seed': 256, 'gamma_Focal': 0,
             'alpha_Balenced': 0, 'aux_littleLabels': False,
              'record_model':True , 'NN_cdf':True}  # 额外参数传递
    opt.parse(extra)

    File_Paths = {'df': "/root/MLabelCls/data/Hashtag_emoji/Reader_Comments.csv" ,
                  'df_kw': '/root/MLabelCls/data/Hashtag_emoji/keywords_CICD.csv',
                  'tokenizer': "/root/Bert_preTrain",
                  'bert': "/root/Bert_preTrain",
                  'trainLM_df':'/root/MLabelCls/data/Hashtag_emoji/Comments_CIDC_trainLM.csv',
                  'model_path': '/Data_storage/model/Bert_SingleLM_reweight.pt', } # Bert_SingleLM_reweight_3， 训练一轮*=/三轮

    if NEED_MODEL_TRAIN:
        train_EmojiCls_MLmodel(File_Paths, opt)

    predictByEmojiClsLM(File_Paths , opt)
    return



# *-----------   三. 花活法  ------------*




if __name__ == '__main__':
    print('论文问题3. 读者端不同方法测试... ')
    # 数据: 评论信息
    # 只是训练技巧/输入与不同模型而已


    # # 1. TextRank Comments:
    # AllIn_keywords()

    # 2. Random_Comments:
    # random_comments()

    # 3. Random SpecificClass_Comments:
    random_specificClass_comments()


    # discreet_Multi_Nominal()
    # discreet_NN()


