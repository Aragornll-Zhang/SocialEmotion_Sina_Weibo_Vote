import torch
import random
import numpy as np
from torch.utils import data
import pandas as pd
from GCN.GCN_dataLoader import *
from transformers import BertTokenizer, BertModel
import pkuseg
from collections import Counter
from sklearn.metrics import confusion_matrix
# check gpu
import pynvml

# 杂活函数们 + config 参数
# ################## *------------------- * #########################
# 1 评价指标:
def evaluate(y , y_pre):
    from sklearn import metrics
    hamming_loss = metrics.hamming_loss(y, y_pre)
    micro_f1 = metrics.f1_score(y, y_pre, average='micro')
    macro_f1 = metrics.f1_score(y, y_pre, average='macro' , zero_division = 1 )
    instance_f1 = metrics.f1_score(y, y_pre, average='samples')
    all_class_f1 = metrics.f1_score(y, y_pre , average=None)
    zero_one_loss = metrics.zero_one_loss(y, y_pre)

    return {'hamming_loss': hamming_loss,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            'instance_f1': instance_f1,
            'zero_one_loss': zero_one_loss,
            'all_class': all_class_f1}


def idx2binary(y_pred , opt):
    # 转换成标签版 binary 矩阵
    N , top_k  = y_pred.shape
    y_pre = np.zeros( (N , opt.label_num ), dtype = int )
    for i in range(N):
        for j in range(top_k):
            if y_pred[i,j] < 0: break
            y_pre[i][ y_pred[i,j] ] = 1
    return y_pre


def oneHot2raw(x):
    x = eval(x)
    assert isinstance(x , (tuple , list) )
    tmp = []
    for i in range(len(x)):
        if x[i]: tmp.append(i)
    return tmp






# ################## *------------------- * #########################
# 2. 随机数
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# # 设置随机数种子
# setup_seed(128)
# print(random.random() , torch.randn(1))

# ################## *------------------- * #########################
# 3. 停用词表
def get_stopWords():
    # 停用词表
    stop_words = set()
    extra = set(['了',' '])
    stop_words |= extra # 空格
    file_path = "/root/wordvc/stop_words_zh.txt"
    with open(file_path , encoding= 'utf-8') as f:
        for elem in f.readlines():
            stop_words.add(elem.strip('\n'))
    print(len(stop_words))
    return stop_words


# ################## *------------------- * #########################
# 4. configs
import warnings
# config
class Bert_Config(object):
    use_cuda = False
    epoch = 9
    batch_size = 16
    learning_rate = 0.0003
    max_grad_norm = 10
    learning_rate_decay =  0.5
    hidden_dim = 768

    If_GCN = True
    GCN_hidden_size = 512
    GCN_emb_size = 768 # 300
    out_feature = 768 # 简单起见，直接等于hidden_dim了
    GCN_layer = 1

    label_num = 24
    User_comments = False
    maxi_len = 23
    maxi_len_with_label = 128

    top_k = 3
    true_Threshold = 0.6 # 设为 predict 标签的阈值
    random_seed = 256 # 全局随机数种子

    # extra loss
    extra_loss = True
    lambda_factor = 0.5
    gamma_Threshold = 0.1

    # NEW 3/1
    Only_Comments = False

    With_abstract = False
    extra_abstract_len = 160

    gamma_Focal = 0
    alpha_Balenced = 0
    aux_littleLabels = False

    record_model = False


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


class RNN_Config(object):
    use_cuda = False
    epoch = 100
    batch_size = 32
    learning_rate = 0.0003
    max_grad_norm = 10
    learning_rate_decay =  0.5
    # RNN parameters
    hidden_dim = 300
    wv_dim = 300 # 300
    cell = 'lstm' # or 'gru'
    num_layers = 1 # rnn 深度
    dropout = 0.2
    bidirectional = True
    # RNN attention
    attention = False
    # GCN
    If_GCN = True
    GCN_hidden_size = 512
    GCN_emb_size = 300
    out_feature = 1024 # 这里就玩个洋的，不跟bert一样了 hidden_dim

    label_num = 24
    User_comments = False
    maxi_len = 14 # max = 14
    maxi_len_with_label = 60 # max: 69 ； 99% <= 61

    top_k = 3
    true_Threshold = 0.6 # 设为predict 标签的阈值
    random_seed =  256 # 全局随机数种子

    # 词向量相关
    vocab_size = 24000

    # 另一个loss
    extra_loss = True
    use_raw_num = 2
    lambda_factor = 0.5

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


# ################## *------------------- * #########################
# 5. split dataset & get_dataset
def split_train_test(data, data2, test_ratio=0.1, valid_ratio = 0.05 ):
  '''
  划分 train.tsv 为训练集/测试集
  :param data1/2:
  :param test_ratio: 划分比例
  :return:
  '''
  assert valid_ratio + test_ratio < 1
  assert len(data) == len(data2)

  # data: pd.dataframe
  np.random.seed(123)
  shuffled_indices = np.random.permutation(len(data))
  test_set_size = int(len(data) * test_ratio )
  valid_size = int(len(data) * valid_ratio )
  test_indices = shuffled_indices[0:test_set_size]
  valid_indices =  shuffled_indices[test_set_size:(test_set_size + valid_size)]
  train_indices = shuffled_indices[(test_set_size + valid_size):]
  return data.iloc[train_indices], data.iloc[valid_indices], data.iloc[test_indices] \
      , data2.iloc[train_indices], data2.iloc[valid_indices], data2.iloc[test_indices]


def get_Bert_dataset(df , tokenizer, df_kw = None , opt = Bert_Config() ):
    text = list( df['hashtag'] )

    if opt.extra_loss:
        raw_labels = list(df['raw_labels'].apply(lambda x: eval(x)))
    else:
        raw_labels = None
    def process_one_hot(x , opt = opt):
        ouput = eval(x)
        one_hot = [0] * opt.label_num
        for label_idx in ouput:
            if label_idx < 0: break
            one_hot[label_idx] = 1
        return one_hot

    labels = list( df['raw_labels'].apply( func = process_one_hot) ) # one-hot
    # print(labels[0] , len(labels))
    if opt.With_abstract:
        abstract_s = list(df['abstract'])
        for i in range(len(text)):
            # text[i] += '[SEP]' + abstract_s[i]
            text[i] = abstract_s[i] # only abstract; 实验用

    if opt.Only_Comments:
        if not opt.User_comments: print('Warning: no inputs now! ')
        text = [''] * len(text)

    if opt.User_comments: # 增加用户评论
        # bert tokenizer 会自动去除空格, 去空格后 mean:79.335 , max = 195 , min 1
        # 上/下十分位 88-73
        key_words = list(df_kw['user_comments'])
        for i in range(len(text)):
            text[i] += '[SEP]' + key_words[i] # .replace(' ', '') # 那我们直接先去一波空格
        opt.maxi_len = opt.maxi_len_with_label

    if opt.With_abstract:
        if opt.User_comments:
            opt.maxi_len = opt.extra_abstract_len + 23 + opt.maxi_len_with_label # extra
        else:
            opt.maxi_len = opt.extra_abstract_len + 23
    all_data = BGCN_DataSet(dat_X= text, tokenizer=tokenizer, raw_labels = raw_labels ,target= labels ,
                            maxi_len= opt.maxi_len , cal_num=3)
    return all_data


def get_RNN_dataset(df , word2idx, seg ,opt , df_kw = None):
    # opt = RNN_Config()
    text = list( df['hashtag'] )
    if opt.extra_loss:
        raw_labels = list(df['raw_labels'].apply(lambda x: eval(x)))
    else:
        raw_labels = None

    def process_one_hot(x , opt = opt):
        ouput = eval(x)
        one_hot = [0] * opt.label_num
        for label_idx in ouput:
            if label_idx < 0: break
            one_hot[label_idx] = 1
        return one_hot

    labels = list( df['raw_labels'].apply( func = process_one_hot) ) # one-hot

    if opt.User_comments: # 增加用户评论
        key_words = list(df_kw['user_comments'])
        for i in range(len(text)):
            text[i] += key_words[i]
        opt.maxi_len = opt.maxi_len_with_label
    Text_idx = [ [ word2idx.get(item, 1) for item in seg.cut(part_txt) ] for part_txt in text ]
    # print(Text_idx[9])
    all_data = RNN_DataSet(dat_X= Text_idx , target= labels, raw_labels = raw_labels , cal_num=2, maxi_len= opt.maxi_len )
    return all_data


# ################## *------------------- * #########################
# 6.  word dictionary For RNN,
def get_corpus(df , df_kw = None , opt = RNN_Config()):
    text = list( df['hashtag'] )
    if opt.User_comments: # 增加用户评论
        # bert tokenizer 会自动去除空格, 去空格后 mean:79.335 , max = 195 , min 1
        # 上/下十分位 88-73
        key_words = list(df_kw['user_comments'])
        for i in range(len(text)):
            text[i] += key_words[i]
        opt.maxi_len = opt.maxi_len_with_label
    return text


def get_wordAndidx(corpus , seg = pkuseg.pkuseg("web") , opt = RNN_Config() ):

    word2idx = {'[PAD]': 0, '[UNK]': 1}
    cnt_dict = Counter()
    for i in range(len(corpus)):
        for item in seg.cut(corpus[i]):
            cnt_dict[item] += 1
    index = len(word2idx)
    for item,_ in cnt_dict.most_common(min(opt.vocab_size - index , len(cnt_dict) ) ):
        word2idx[item] = index
        index += 1
    assert index == len(word2idx)
    return word2idx

# new loss
class AuxLoss(nn.Module):
    def __init__(self, top_k = 2, weight = None , device = 'cpu' ,
                 alpha_Belance = 0 , gamma_Focal = 0 ):
        super(AuxLoss, self).__init__()
        if isinstance(weight , torch.Tensor):
            self.weight = weight # 指定权重 // 后续甚至可学习
        else:
            self.weight = torch.ones( top_k , requires_grad=False) / top_k # 固定weights
        self.weight = self.weight.to(device)
        self.device = device
        self.top_k = top_k

        # Belanced-CE & Focal loss 3/31
        self.alpha_Belance = alpha_Belance
        self.gamma_Focal = gamma_Focal


    def forward(self, y_pred , target , aux_target = None):
        '''
            改进版 CrossEntropy,
        :param y_pred: [batch size , label_num] , 范围: [-\infty , \infty]
        :param target: [batch , raw labels (After Padding)] e.g. top 2/3 emojis In This Multi-label cls
        :param aux_target: [label_num] , 为Balenced-CE 记录哪些小样本需要额外加权
        :return:  loss
        '''

        bz, N = target.shape
        if N > self.top_k:
            target = target[:, 0:self.top_k]

        masks = self.get_masked(target)
        y_pred = torch.softmax(y_pred , dim=-1) # involution + exp

        Lower_bound = torch.tensor([-100.0]).to(self.device)
        # for 循环大法
        for i in range(bz):
            tmp = y_pred[i, :]
            true_target_len = masks[i].sum().item()
            if true_target_len == self.top_k:
                discount = 1
            else:
                discount = 1 - self.weight[self.top_k - true_target_len : ].sum().item()

            if self.alpha_Belance == 0 and self.gamma_Focal == 0:
                # 是否应该关心数值稳定性 + torch.max(loss , -100) ?
                if i == 0:
                    loss = -torch.max( torch.log( torch.dot( tmp[target[i]] * masks[i] , self.weight) / discount ) , Lower_bound )
                else:
                    loss += -torch.max( torch.log( torch.dot( tmp[target[i]] * masks[i] , self.weight) / discount ) , Lower_bound ) # torch.dot( tmp[target[i]] , self.weight)
            elif self.alpha_Belance > 0: # and aux_target is not None:
                # assert (aux_target is not None)
                # aux_target e.g. [0,0,0,...,1,1,1,1] 数量的少的标签，单拿出来给加点权重
                if aux_target is not None:
                    if not isinstance(aux_target, torch.Tensor):
                        aux_target = torch.tensor(aux_target)
                    tmp_aux = aux_target[target[i]].to(self.device)
                    focal_coeff = (1 - tmp[target[i]]) ** self.gamma_Focal  # 取0时，直白focal loss没影响
                    Bi_coeff = focal_coeff * (  (1-self.alpha_Belance)*(1-tmp_aux)  +  self.alpha_Belance*tmp_aux )
                    if i == 0:
                        loss = -torch.max( torch.log( torch.dot( tmp[target[i]] * masks[i] * Bi_coeff , self.weight) / discount ) , Lower_bound )
                    else:
                        loss += -torch.max( torch.log( torch.dot(  tmp[target[i]] * masks[i] * Bi_coeff , self.weight) / discount ) , Lower_bound )
                else:
                    if i == 0:
                        loss = -torch.max(torch.log(torch.dot(tmp[target[i]] * masks[i], self.weight) / discount), Lower_bound)
                    else:
                        loss += -torch.max(torch.log(torch.dot(tmp[target[i]] * masks[i], self.weight) / discount), Lower_bound)
            else:
                # only focal loss
                focal_coeff = (1 - tmp[target[i]]) ** self.gamma_Focal
                if i == 0:
                    loss = -torch.max( torch.log( torch.dot( focal_coeff * tmp[target[i]] * masks[i] , self.weight) / discount ) , Lower_bound )
                else:
                    loss += -torch.max( torch.log( torch.dot( focal_coeff * tmp[target[i]] * masks[i] , self.weight) / discount ) , Lower_bound )

        return loss / bz # 平均loss


    def get_masked(self, target):
        '''
        :param target: []
        :return:
        '''
        return target.ne(-1).to(self.device) # -1 padding when generating dataloader




def test_AUXLOSS():
    import torch
    import torch.nn as nn
    # nn.CrossEntropyLoss()
    loss_func = AuxLoss()


    def test():
        y_pred = torch.randn((32,5))
        target = torch.empty((32,5), dtype=torch.long).random_(5)
        loss_func = AuxLoss()
        loss = loss_func(y_pred , target)
        print(loss)

        m = nn.CrossEntropyLoss()
        output = m(y_pred, target[:,0])
        print(output)



def Spare_gpu(index = (0,1,2,3) ):
    pynvml.nvmlInit()
    maxi = None
    best = index[0]
    for idx in index:
        h = pynvml.nvmlDeviceGetHandleByIndex(idx)
        m = pynvml.nvmlDeviceGetMemoryInfo(h)
        spare = m.free / m.total
        if maxi:
            if spare > maxi:
                best = idx
                maxi = spare
        else:
            maxi = spare
            best = idx
    return best , maxi


# test pytorch gradient backpropagation aux loss
def test():
    import torch
    import numpy as np
    import torch.nn as nn

    input_np = np.random.randn(16,10)
    target = torch.empty(16, dtype=torch.long).random_(5)
    target_aux = torch.ones([16, 5], dtype=torch.float32)

    input = torch.tensor(input_np, requires_grad=True , dtype=torch.float32)
    model = nn.Sequential( nn.Linear(10, 10), nn.ReLU() , nn.Linear(10 , 5) )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001,)

    loss_func = nn.CrossEntropyLoss()
    loss_func_aux = nn.BCEWithLogitsLoss()

    model.zero_grad()
    output = model(input)

    loss = loss_func(output , target)
    loss_aux = loss_func_aux(output , target_aux)

    alpha = 0.9
    loss_ = alpha *10000* loss + (1-alpha) * loss_aux
    print(loss_)
    loss_.backward()

    # 查看梯度
    for name , blk in model.named_parameters():
        print(name)
        print(blk.grad)
        break


    # Test BCE
    x = torch.tensor([[0.2,0.1,0.15,0.15,0.4] , [0.2,0.1,0.3,0.1,0.3]])
    multi_target = torch.tensor([[0,0,1,1,1],[1,0,0,1,0]] , dtype=torch.float32)

    loss_func = nn.BCELoss()
    loss = loss_func(x , multi_target)

    loss_b = multi_target * torch.log(x) + (1-multi_target) * torch.log(1-x)
    print( loss_b.sum() / 10 )



