import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer
import warnings
from torch.utils import data
import pickle
from GCN.util import AuxLoss
from RC_CogLTX.CogLTX_buffer import Block , Buffer

DEFAULT_MODEL_NAME = '/root/Bert_preTrain'

# *--------   1. config  ---------------*
class Judge_Config(object):
    use_cuda = False
    epoch = 9
    batch_size = 16
    learning_rate = 1e-4
    max_grad_norm = 10
    learning_rate_decay = 0.5
    hidden_dim = 768

    label_num = 2
    maxi_len = 100 # 平均评论长度
    maxi_len_with_label = 100
    use_hashtag = False # True: [cls]#一眼丁真#[sep]comment_{z1}[sep]comment_{z_2}...

    shuffle_dl = True # dataLoader shuffle

    top_k = 3
    random_seed = 256 # 全局随机数种子

    # extra loss
    train_model_name = 'judge'
    extra_loss = True # 无监督中的相关性
    lambda_factor = 0.5 # extra_loss 占比

    # bert model 参数

    PLM_name = '/root/Bert_preTrain'
    save_path = '/Data_storage/model/cogLTX/judge_LM.pt'

    def parse(self, kwargs):
        '''
        根据字典kwargs 更新 config参数
        '''
        # 更新配置参数
        for k, v in kwargs.items():
            if not hasattr(self, k):
                # 警告还是报错，取决于你个人的喜好
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)
        return


class Reanson_Config(object):
    use_cuda = False
    epoch = 9
    batch_size = 16
    learning_rate = 5e-5
    max_grad_norm = 10
    learning_rate_decay = 0.5
    hidden_dim = 768

    # GCN
    If_GCN = True
    GCN_hidden_size = 512
    GCN_emb_size = 768
    out_feature = 768 # 简单起见，直接等于hidden_dim了
    GCN_layer = 1
    gamma_Threshold = 0.1

    label_num = 24
    maxi_len = 160 # 256 # 压缩后
    use_hashtag = False # True: [cls]#一眼丁真#[sep]comment_{z1}[sep]comment_{z_2}...
    threshold_up = 0.2
    threshold_down = -0.1

    shuffle_dl = True # dataLoader shuffle

    top_k = 3
    random_seed = 256 # 全局随机数种子
    extra_loss = False

    # extra loss, e.g. focal loss
    gamma_Focal = 0
    alpha_Balenced = 0

    # For Train
    train_model_name = 'reasoner'
    # bert model 参数
    PLM_name = '/root/Bert_preTrain'


    def parse(self, kwargs):
        '''
        根据字典kwargs 更新 config参数
        '''
        # 更新配置参数
        for k, v in kwargs.items():
            if not hasattr(self, k):
                # 警告还是报错，取决于你个人的喜好
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)
        return


# *-------- 2. data , dataset & dataloader  --------*
def get_SingleHashtag_corpus( file_idx ):
    save_path = '/Data_storage/CICD_merge/{}.txt'
    Comments = []
    Rele_labels = []
    with open(save_path.format(file_idx) , 'r' , encoding='utf-8') as f:
        first_row = True
        for line in f.readlines():
            if first_row:
                first_row = False
            else:
                # try:
                str_list = line.strip('\n').split('\t')
                comment, rele_label = ''.join(str_list[0:-1]) , str_list[-1]
                # except:
                #     print(line)
                #     assert False

                Comments.append(comment)
                Rele_labels.append(int(rele_label) )
    return Comments , Rele_labels
# a , b =get_SingleHashtag_corpus(0)
# print(len(a) , len(b) , b[0:5])


class Judge_DataSet(data.Dataset):
    def __init__(self, dat_X, target , target_aux = None , opt = Judge_Config()):
        self.dat = dat_X
        self.target = target # 0/1 二分类
        self.target_aux = target_aux
        self.opt = opt
        return

    def __getitem__(self, index):
        '''
        :param index:
        :return:
        '''
        # sentence ： [101 开始符, xxx ,xx ,x ,x , 0 , 0 ,... , 102 终止符]
        maxi_len = self.opt.maxi_len
        sentence = self.dat[index]
        N = len(sentence)
        attn_mask = torch.zeros( maxi_len , dtype=torch.long )
        attn_mask[0: min( N , maxi_len ) ] = 1
        if N < maxi_len:
            sentence = sentence + [0] * (maxi_len - N)
        return torch.tensor(sentence[0:maxi_len] , dtype=torch.long), \
               attn_mask, torch.tensor(self.target[index], dtype=torch.long) , \
               torch.tensor(self.target_aux[index], dtype=torch.long)

    def __len__(self):
        return len(self.dat)


class Reason_DataSet(data.Dataset):
    def __init__(self, source):
        self.dat = source
        return

    def __getitem__(self, index):
        '''
        :param index:
        :return:
        '''
        # sentence ： [101 开始符, xxx ,xx ,x ,x , 0 , 0 ,... , 102 终止符]
        return self.dat[index]

    def __len__(self):
        return len(self.dat)



def get_judge_dateloader( Buffer_list, opt = Judge_Config() ):
    All_Corpus, target_1, target_2 = [],[],[]
    for bur in Buffer_list:
        c ,t1 ,t2 = bur.export_blocks()
        All_Corpus += c # bur
        target_1 += t1 # 自带 relevant
        target_2 += t2 # loss 学出的 relevant
    dataset = Judge_DataSet(All_Corpus, target_1 , target_aux=target_2 , opt=opt)
    return data.DataLoader(dataset, batch_size= opt.batch_size, shuffle=True, drop_last=False )


def get_reasoner_dataloader( Buffer_list, Target_labels ,opt = Reanson_Config()):
    source = []
    for i , bur in enumerate(Buffer_list):
        dat_X , attn_mask = bur.export(drop_idx_set={-1} , total_length=opt.maxi_len)
        source.append( (dat_X , attn_mask, torch.tensor(Target_labels[i] , dtype=torch.long) ) )
    dataset = Reason_DataSet(source)
    dl = data.DataLoader(dataset , batch_size=opt.batch_size , shuffle=True , drop_last=False)
    return dl



# *--------- 3. 计算全局 score -------*
def calc_estimation_score( Buffer_list, intros_model, opt = Judge_Config() , device = 'cpu'):
    with torch.no_grad():
        for bur in Buffer_list:
            comments , t1 , t2 = bur.export_blocks()
            tmp_dataset = Judge_DataSet( comments, t1, target_aux=t2, opt=opt)
            dl = data.DataLoader(tmp_dataset , batch_size= len(comments), shuffle=False, drop_last=False)
            for i, batch in enumerate(dl): # 仅一个batch
                x , mask = batch[0:2]
                x = x.to(device)
                mask = mask.to(device)
                output = intros_model(x , mask) # [] TODO: relevance 分数, 譬如 softmax
                scores = torch.softmax(output ,dim=-1)[:, 1]

            assert scores.shape[0] >= (len(bur) -1)
            for j in range( scores.shape[0] ): # 赋值
                # bur = Buffer()
                bur.blocks[j].estimation = scores[j].item() # ???
                # print('xxx')
            # break
    return


def calc_rele_acquired( Buffer_list, reason_model, Y, loss_func = nn.CrossEntropyLoss(), opt = Reanson_Config() , device = 'cpu' ):
    # new Buffer_list Z
    # 这一步或许会非常慢, 不过与 calc estimation 相比能快不少
    T_up = opt.threshold_up
    T_down = opt.threshold_down
    Up_cnt = 0
    Down_cnt = 0
    Total = 0
    with torch.no_grad():
        for index , bur in enumerate(Buffer_list):
            # bur = Buffer()
            ids , attn_mask = bur.export( total_length=opt.maxi_len , drop_idx_set={-1} )
            ids , attn_mask = ids.view(1,-1).to(device), attn_mask.view(1,-1).to(device)
            output = reason_model(ids , attn_mask)
            y = torch.tensor( Y[index] , dtype=torch.long ).view(1, -1) # .to(device)
            loss_origin = loss_func(output, y)
            Total += len(bur)
            for i in range(1, len(bur)//3 ): # Notice: 默认有 hashtag_head
                if len(bur.blocks[i]) >= opt.maxi_len // 3:
                    extra = (len(bur.blocks[i]) / opt.maxi_len) * 0.15
                else:
                    extra = 0
                _ids , _attn_mask = bur.export( total_length=opt.maxi_len , drop_idx_set={i} )
                _ids, _attn_mask = _ids.view(1, -1).to(device), _attn_mask.view(1, -1).to(device)
                _output = reason_model(_ids , _attn_mask)
                loss_ = loss_func(_output, y)
                # 根据 loss_ - loss_origin 设置 relevance
                if loss_ - loss_origin >= T_up + extra:
                    bur.blocks[i].relevance_acquired = 1
                    Up_cnt += 1
                    # print('T_up changed relevant! ')
                elif loss_ - loss_origin <= T_down - extra:
                    bur.blocks[i].relevance_acquired = 0
                    Down_cnt += 1
                    # print('T_down changed irrelevant! ')
            # break
    print('Ratio , Up : {}  and Down : {}'.format(  Up_cnt/max(Total , 1) ,  Down_cnt/max(Total , 1)) )
    return


# *--------- 4. 获取 Buffer / Buffer_List  -------*
def construct_buffer(Corpus, tokenizer, RelevanceList = None):
    '''
    :param Corpus: 类似列表的一个东西
    :param tokenizer:
    :param RelevanceList: 天然相关性; 譬如，是否在top3类中
    :return:
    '''
    block_list = []
    hashtag_head = True # TODO: 目前直白全加 hashtag 头， 未来可整花活
    for i , comment in enumerate(Corpus):
        encode_dict = tokenizer.encode_plus(comment, add_special_tokens=True)  # i.e. padding = False
        ids = encode_dict['input_ids'] # e.g. [101, 2303, 2094, 2643, 102]
        block_list.append( Block(ids=ids,pos=i, relevance=RelevanceList[i]) )
    return Buffer(block_list , hashtag_head)


def get_BufferList(df , tokenizer):
    Buffer_list = []
    Rele_rate = 0
    length = 0
    for i in range(len(df)):
        file_idx = df['abstract_df_idx'].iloc[i]
        tmp_corpus, relevant_list = get_SingleHashtag_corpus(file_idx)  # 譬如,读文件法  return: A list or sth
        # 加入 Hashtag 信息
        tmp_corpus = [df['hashtag'].iloc[i]] + tmp_corpus
        relevant_list = [1] + relevant_list
        buffer = construct_buffer(Corpus=tmp_corpus, tokenizer=tokenizer, RelevanceList=relevant_list)
        Buffer_list.append(buffer)

        Rele_rate += sum(relevant_list) - 1
        length += len(relevant_list) - 1
    print('Rele_rate: ' ,  Rele_rate / length)
    return Buffer_list






