import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from GCN.Graph_init import  *



# embedding for GCN
def get_label_embedding(label_num , emb_size = 64, method = 'random',
                        model = None , tokenizer = None, device = 'cpu' , mapping = idx2emoji ):
    if method == 'random':
        embedding = nn.Embedding(label_num , emb_size)
        return embedding(torch.LongTensor([i for i in range(label_num)]) )
    elif method == 'bert_encoder':
        # TODO: 写一个 embedding
        # tokenizer = BertTokenizer.from_pretrained("F:\\SentimentCls\\Bert预训练")
        # model = BertModel.from_pretrained("F:\\SentimentCls\\Bert预训练")
        for i in range(label_num):
            input = torch.tensor(tokenizer.encode( mapping[i] )[1:-1] ).unsqueeze(0).to(device)
            # embedding_output = model.embeddings( input_ids= input)
            encoding = model.embeddings( input_ids= input) # model( input )[1] # dim: 768
            encoding = encoding.sum(dim=1) / len(input)
            if i == 0:
                out = encoding
            else:
                out = torch.cat([out , encoding] , dim=0)
        assert out.shape[0] == label_num
        return out # 也可再加一波线性层转维度
    elif method == 'word_vc':
        save_path = '/root/MLabelCls/data/label_emb/emoji_24_emb.npy'
        emb_s = np.load(save_path)
        return torch.tensor(emb_s , dtype=torch.float32)
    else:
        raise NotImplementedError


def word_vec():
    import numpy as np
    # https://github.com/Embedding/Chinese-Word-Vectors  # weibo , 吃水不忘挖井人
    file_path = '/root/wordvc/weibo_wv/sgns.weibo.bigram-char'
    emoji_phrases = ['心', '费解', '狗头', ('捂脸', '哭笑不得'), '泪', '赞', '怒', '憧憬', ('笑', '哭笑不得'), '旁观', '蜡烛', '微笑', '伤心', '牛人', ('中国','赞'), '吃惊', '跪', ('给你个', '心'), '嘻嘻', '拜拜', '酸', '暗藏玄机', '思考', '威武']
     #    {'心': 0, '费解': 1, '狗头': 2, ('捂脸', '哭笑不得'): 3, '泪': 4, '赞': 5, '怒': 6, '憧憬': 7, ('笑', '哭笑不得'): 8, '吃瓜': 9, '蜡烛': 10,
     # '微笑': 11, '伤心': 12, '给力': 13, '中国赞': 14, '吃惊': 15, '跪了': 16, ('给你个', '心'): 17, '嘻嘻': 18, '拜拜': 19, '酸': 20,
     # '并不简单': 21, '思考': 22, '威武': 23}
    record_wv = np.zeros( (24,300) , dtype=float )
    for i in range(len(record_wv)):
        target = emoji_phrases[i]
        with open(file_path , 'r' , encoding='utf-8') as f:
            first_row = True
            cnt = 0
            for line in f.readlines():
                if first_row:
                    first_row = False
                    continue
                line = line.strip('\n')
                tmp = line.strip(' ').split(' ')
                phrase = tmp[0]
                if isinstance(target , str):
                    if phrase == target:
                        try:
                            tmp_wv = list(map(float, tmp[1:]))
                            record_wv[i] = np.array( tmp_wv )
                            break
                        except:

                            print('Wrong here , ' , i, target, tmp[1:])

                else:
                    if phrase in target:
                        try:
                            tmp_wv = list(map(float, tmp[1:]))
                            tmp_wv = np.array(tmp_wv)
                            if cnt == 0:
                                record_wv[i] = tmp_wv
                            else:
                                record_wv[i] += tmp_wv
                            cnt += 1
                            if cnt == len(target):
                                record_wv[i] = record_wv[i] / cnt
                                break
                        except:
                            print(i , target , tmp[1:])


            if record_wv[i][0] == 0:
                print(target)

    # print(record_wv)
    save_path = '/root/MLabelCls/data/label_emb/emoji_24_emb.npy'
    np.save(save_path , record_wv)
    print('已保存!')
    # print( np.load(save_path) )
    return


# dataset & dataloader
class BGCN_DataSet(data.Dataset):
    def __init__(self, dat_X, tokenizer, maxi_len = 23,raw_labels = None , target=None, cal_num = 1):
        self.dat = dat_X
        self.target = target
        self.max_sentence_len = maxi_len
        self.tokenizer = tokenizer
        self.raw_target = raw_labels
        self.cal_num_for_Raw = cal_num
        return

    def __getitem__(self, index):
        '''
        :param index:
        :return:
        '''
        # sentence ： [101 开始符, xxx ,xx ,x ,x , 102 终止符, 0 , 0 ,... , ]
        encode_dict = self.tokenizer.encode_plus(self.dat[index] , truncation=True , add_special_tokens=True , max_length=self.max_sentence_len, padding='max_length' )
        sentence = encode_dict['input_ids']
        mask = encode_dict['attention_mask']
        if self.raw_target is not None:
            raw_label = self.raw_target[index]
            if self.cal_num_for_Raw <= len(raw_label):
                raw_label = raw_label[0:self.cal_num_for_Raw]
            else:
                tmp_n = len(raw_label)
                raw_label = raw_label + [-1] * ( self.cal_num_for_Raw - tmp_n )   # use -1 to Pad

            return torch.tensor(sentence , dtype= torch.long ) , torch.tensor(mask) ,  \
                   torch.tensor( self.target[index] , dtype=torch.float32) , torch.tensor( raw_label )
        else:
            return torch.tensor(sentence), torch.tensor(mask), \
            torch.tensor(self.target[index], dtype=torch.float32)

    def __len__(self):
        return len(self.dat)


class RNN_DataSet(data.Dataset):
    def __init__(self, dat_X, maxi_len = 23, target=None, raw_labels = None , cal_num = 1):
        self.dat = dat_X
        self.target = target
        self.max_sentence_len = maxi_len
        self.raw_target = raw_labels
        self.cal_num_for_Raw = cal_num
        return

    def __getitem__(self, index):
        '''
        :param index:
        :return:
        '''
        sentence = self.dat[index]
        N = len(sentence)
        if len(sentence) > self.max_sentence_len:
            sentence = sentence[0:self.max_sentence_len]
        elif len(sentence) < self.max_sentence_len:
            pad_len = self.max_sentence_len - len(sentence)
            sentence = sentence + [0] * pad_len
        if self.raw_target: # not None
            raw_label = self.raw_target[index]
            if self.cal_num_for_Raw <= len(raw_label):
                raw_label = raw_label[0:self.cal_num_for_Raw]
            else:
                tmp_n = len(raw_label)
                raw_label = raw_label + [-1] * ( self.cal_num_for_Raw - tmp_n )   # using -100 Padding
            # raw_label = raw_label[0: min(self.cal_num_for_Raw, len(raw_label))]
            return torch.tensor(sentence) , min(N ,self.max_sentence_len) \
                ,torch.tensor( self.target[index] , dtype=torch.float32) , torch.tensor( raw_label )
        else:
            return torch.tensor(sentence) , min(N ,self.max_sentence_len) \
                ,torch.tensor( self.target[index] , dtype=torch.float32)



    def get_raw_labels(self , one_hot):
        assert isinstance(one_hot, (tuple, list))
        tmp = []
        for i in range(len(one_hot)):
            if one_hot[i]: tmp.append(i)
        return tmp

    def __len__(self):
        return len(self.dat)


def collate_fn_rnn(data_batch):
    # data_batch.sort(key=lambda x: x[1], reverse=True)
    data = [s[0] for s in data_batch]
    seq_len = [s[1] for s in data_batch]
    target = [ s[-1]  for s in data_batch]
    Maxi = max(seq_len)
    return ( ( torch.tensor(data)[: , 0:Maxi ] , seq_len ) ,torch.tensor(target))




def drawHist():
    # 7 ~ 15 词比较多 max:21
    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.read_csv("F:\specific_python\FinalGame\SGM\HashtagAndEmoji.csv", usecols=['hashtag'])
    record = np.array(list(df['hashtag'].apply(lambda x: len(x))))
    plt.hist(record)
    plt.show()

def testUpdate():
    print('3')

# if __name__ == '__main__':
#     # testUpdate()
#     # word_vec()
#     # import  numpy as np
#     # save_path = '/root/MLabelCls/data/label_emb/emoji_24_emb.npy'
#     # a = np.load(save_path)
#     # print(a[:, 0])