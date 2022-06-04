import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import BertModel, BertTokenizer
from torch.utils import data


from GCN.Bert_GCN import Bert_GCN
from GCN.util import Spare_gpu, split_train_test, confusion_matrix , evaluate, idx2binary, AuxLoss


def get_CommentFromCIDC_rawLabel(file_idx , Prior, Maxi_num = 10):
    '''   用来挑选特殊的应景 comments 来进行训练
    :param file_idx:
    :param Prior: 所需要类型
    :param Maxi_num:
    :return:
    '''
    save_path = '/Data_storage/CICD_rawLabel/{}.txt'
    Comments = [ [] for _ in range(len(Prior)) ]
    Prior = { tuple(sorted((k))) : v for k , v in Prior.items() }
    with open(save_path.format(file_idx) , 'r' , encoding='utf-8') as f:
        first_row = True
        for line in f.readlines():
            if first_row:
                first_row = False
            else:
                str_list = line.strip('\n').split('\t')
                comment, rele_label = ''.join(str_list[0:-1]) , str_list[-1]
                # try:
                rele_label = tuple(sorted((eval(rele_label))))
                # except:
                #     print('here')
                if rele_label in Prior:
                    idx = Prior[rele_label]
                    try:
                        if len(Comments[idx]) < Maxi_num  and (2 <= len(comment) <= 60):
                            Comments[idx].append( comment )
                    except:
                        print(Prior)
                        print(rele_label , idx, len(Comments) , file_idx )
                        assert False
    return Comments


def get_SingleLabelFromCIDC_rawLabel(file_idx):
    save_path = '/Data_storage/CICD_rawLabel/{}.txt'
    Comments = [] # comment And Label
    Labels = []
    with open(save_path.format(file_idx) , 'r' , encoding='utf-8') as f:
        first_row = True
        for line in f.readlines():
            if first_row:
                first_row = False
            else:
                str_list = line.strip('\n').split('\t')
                comment, rele_label = ''.join(str_list[0:-1]) , str_list[-1]
                Comments.append(comment)
                Labels.append(rele_label)
    return Comments, Labels


# *-----------  训练单个语言模型所用  ---------------*
class SingleML_DataSet(data.Dataset):
    def __init__(self, dat_X, tokenizer, opt ,target=None):
        self.dat = dat_X
        self.target = target
        self.opt = opt
        self.max_sentence_len = opt.maxi_len
        self.tokenizer = tokenizer
        return

    def __getitem__(self, index):
        '''
        :param index:
        :return:
        '''
        # sentence ： [101 开始符, xxx ,xx ,x ,x , 0 , 0 ,... , 102 终止符]
        # tokenizer = BertTokenizer.from_pretrained("F:\\SentimentCls\\Bert预训练")  # path or name;
        # self.tokenizer = tokenizer
        encode_dict = self.tokenizer.encode_plus(self.dat[index] ,padding = 'max_length' ,truncation=True , add_special_tokens=True , max_length=self.max_sentence_len )
        sentence = encode_dict['input_ids']
        mask = encode_dict['attention_mask']
        # sentence = self.tokenizer.encode( self.dat[index] , add_special_tokens=True , max_length=self.max_sentence_len ,pad_to_max_length=True)
        if self.target is not None:
            return torch.tensor(sentence) , torch.tensor(mask),  torch.tensor( self.target[index] , dtype=torch.long)
        else:
            return torch.tensor(sentence ) , torch.tensor(mask)

    def __len__(self):
        return len(self.dat)


def train_EmojiCls_MLmodel(File_Paths , opt , MODE=0):

    CURRENT_SPARE_GPU , free_rate = Spare_gpu((0,1,2))  # 2
    print("gpu_id {} , free_rate:{} ".format( CURRENT_SPARE_GPU , free_rate) )
    if free_rate <= 0.5:
        print('gpu 内存或许不足！')
        assert False

    if torch.cuda.is_available() and opt.use_cuda:
        if MODE == 0: # for SSH
            device = torch.device("cuda:{}".format(CURRENT_SPARE_GPU) if torch.cuda.is_available() else "cpu")
        else: # for colab
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )
    else:
        device = 'cpu'

    Model_PATH = File_Paths['model_path']
    tokenizer = BertTokenizer.from_pretrained(File_Paths['tokenizer']) # "/root/Bert_preTrain"
    df = pd.read_csv( File_Paths['trainLM_df'], encoding='utf-8')
    train_df, valid_df,test_df,_,_,_ = split_train_test(df , df.copy() , test_ratio=0.15 , valid_ratio=0.001 )

    # 手动均衡化 train_df:
    train_df.sample(frac=1)
    record_idx = []
    record_num = {}
    for i in range(len(train_df)):
        label_idx = df['label'].iloc[i]
        if label_idx in record_num:
            if record_num[label_idx] < 2500:
                record_num[label_idx] += 1
                record_idx.append(i)
        else:
            record_num[label_idx] = 0
    train_df = train_df.iloc[record_idx]

    train_set = SingleML_DataSet(dat_X=list(train_df['corpus']), tokenizer=tokenizer, target= list(train_df['label']) ,opt=opt )
    train_dl = data.DataLoader(train_set, batch_size=opt.batch_size, drop_last=False, shuffle=True)

    test_set = SingleML_DataSet(dat_X=list(test_df['corpus']), tokenizer=tokenizer, target= list(train_df['label']),opt=opt)
    test_dl = data.DataLoader(test_set, batch_size=opt.batch_size, drop_last=False, shuffle=True)

    bert_model = BertModel.from_pretrained(File_Paths['bert'])
    model = Bert_GCN(bert_model , opt)

    print('get here!', device)
    model.to(device)
    print('model to {} ... '.format(device))
    optimizer = torch.optim.Adam( model.parameters(), lr = opt.learning_rate , weight_decay = opt.learning_rate_decay )
    loss_func = nn.CrossEntropyLoss()


    # Train Model
    last_acc = None
    for epoch in range(opt.epoch):
        print('*----------------------------------*')
        print('new Epoch {}'.format(epoch))
        print('*----------------------------------*')
        model.train()
        total_train_loss = 0
        C = np.array([[0] * opt.label_num for _ in range(opt.label_num)])
        # train
        for i, batch in enumerate(train_dl):
            x, mask, y = batch
            input_x = x.to(device)
            mask = mask.to(device)
            y = y.to(device)
            optimizer.zero_grad()

            output = model(input_x,mask)
            loss = loss_func(output, y)

            # backward and optimize
            total_train_loss += loss.item()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 防止梯度爆炸
            optimizer.step()

            # 评估
            y_pred = np.argmax(output.detach().cpu().numpy(), axis=1).flatten()
            y_true = y.to('cpu').numpy()
            tmp_C = confusion_matrix(y_true=y_true, y_pred=y_pred , labels=[_ for  _ in range(opt.label_num)])
            C += tmp_C

        print('train loss:', total_train_loss / len(train_dl))  # 那一点小偏差无所谓
        print('Train: accuracy: ', C.trace() / C.sum())

        # test dataSet; validation
        model.eval()
        C = np.array([[0] * opt.label_num for _ in range(opt.label_num)])
        with torch.no_grad():
            for i, batch in enumerate(test_dl):
                x, mask, y = batch
                input_x = x.to(device)
                mask = mask.to(device)
                # y = y.to(device)
                output = model(input_x,mask)
                # 评估
                y_pred = np.argmax(output.detach().cpu().numpy(), axis=1).flatten()
                y_true = y.to('cpu').numpy()
                tmp_C = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[_ for  _ in range(opt.label_num)])
                C += tmp_C

        print('Test: accuracy: ', C.trace() / C.sum())
        if (last_acc is None) or (last_acc <= C.trace() / C.sum()):
            if epoch > 0: last_acc = C.trace() / C.sum()
            torch.save(model.state_dict(), Model_PATH)
        elif epoch > 3:
            break

    print('End here!!!')
    return



class Simple_dataset(data.Dataset):
    def __init__(self, source):
        self.dat = source
        return
    def __getitem__(self, index):
        return self.dat[index]
    def __len__(self):
        return len(self.dat)



def get_dataset_For_NN_Cdf(df,model, tokenizer,opt , Seq_Len = 64, device ='cpu'):
    # seq_len = 48  # 48 条评论
    Source = []
    with torch.no_grad():
        for i in range(len(df)):
            file_idx = df['abstract_df_idx'].iloc[i]
            hashtag = df['hashtag'].iloc[i]
            raw_labels = eval( df['raw_labels'].iloc[i] )
            Comments, _ = get_SingleLabelFromCIDC_rawLabel(file_idx)
            Comments = [hashtag + '[SEP]' + c for c in Comments]
            single_dataset = SingleML_DataSet(Comments, tokenizer, opt)
            tmp_dl = data.DataLoader(single_dataset, batch_size=opt.batch_size, drop_last=False, shuffle=True)

            for j, batch in enumerate(tmp_dl):
                x, mask = batch
                input_x = x.to(device)
                mask = mask.to(device)

                output = model(input_x, mask)  # [小bz, 24 ]
                if j == 0:
                    same_comment_output = output.detach()
                else:
                    same_comment_output = torch.cat( (same_comment_output, output.detach()) )
                if same_comment_output.shape[0] >= Seq_Len:
                    same_comment_output = same_comment_output[0:Seq_Len]

            tmp_seq_len = Seq_Len
            if same_comment_output.shape[0] < Seq_Len:
                tmp_seq_len = same_comment_output.shape[0]
                same_comment_output = F.pad( same_comment_output , [0,0,0, Seq_Len-tmp_seq_len])
            Source.append( (same_comment_output , tmp_seq_len, torch.tensor(raw_labels , dtype=torch.long )  ) )
    dataset = Simple_dataset(source=Source)
    return dataset


# *----------- 拟合概率分布：全连接 + Relu  -----------------*
class NN_ForCdf(nn.Module):
    def __init__(self, input_size=24, hidden_size=64, output_size=24 ,layer_num=2, ):
        super(NN_ForCdf, self).__init__()
        f_list = []
        if layer_num == 1:
            f_list.append(nn.Linear(in_features=input_size, out_features=output_size ))
        else:
            f_list.append(nn.Linear(in_features=input_size, out_features=hidden_size ) )
            f_list.append(nn.ReLU())
        for _ in range(1, layer_num):
            if _ == layer_num-1:
                f_list.append(nn.Linear(in_features=hidden_size, out_features=output_size , bias=False))
            else:
                f_list.append(nn.Linear(in_features=hidden_size, out_features=hidden_size ))
                f_list.append( nn.ReLU() )
        self.f = nn.Sequential(*f_list)


    def forward(self, x , seq_len ):
        output = self.f(x)
        seq_len = seq_len.float() # [ bz, raw_label ]
        output = ( output.sum(dim=1).transpose(0,1) / seq_len ).transpose(0,1) # [ Batch size , 24 ]
        return output



def predictByEmojiClsLM(File_Paths , opt , MODE=0):
    print('start prediction...')

    tokenizer = BertTokenizer.from_pretrained(File_Paths['tokenizer']) # "/root/Bert_preTrain"

    df = pd.read_csv( File_Paths['df'], encoding='utf-8')
    # df = df.iloc[0:100]
    train_df,valid_df,test_df,_,_,_ = split_train_test(df , df.copy()  )
    # df = test_df

    bert_model = BertModel.from_pretrained(File_Paths['bert'])
    model = Bert_GCN(bert_model , opt)
    # 利用训好的模型参数
    Model_PATH = File_Paths['model_path']
    model.load_state_dict(torch.load(Model_PATH))

    CURRENT_SPARE_GPU, free_rate = Spare_gpu((0, 1, 2))  # 2
    print("gpu_id {} , free_rate:{} ".format(CURRENT_SPARE_GPU, free_rate))
    if torch.cuda.is_available() and opt.use_cuda:
        if MODE == 0: # for SSH
            device = torch.device("cuda:{}".format(CURRENT_SPARE_GPU) if torch.cuda.is_available() else "cpu")
        else: # for colab
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )
    else:
        device = 'cpu'

    model.to(device)
    model.eval()

    if opt.NN_cdf: # 额外训练分布, 先练个概率分布
        # input : [batch_size , sequence_len(comments_num) , label_num(24) ] -> output: [batch size, label_num(24)]
        # TODO 调参
        model_cdf = NN_ForCdf(input_size=opt.label_num, hidden_size=64, layer_num=2, output_size= opt.label_num)
        model_cdf.to(device)
        # Seq_Len: Comments num
        data_set = get_dataset_For_NN_Cdf(df=train_df, model=model ,tokenizer=tokenizer,opt=opt, Seq_Len =48,device =device)
        train_dataset, valid_dataset = data.random_split( data_set , [int(0.8 * len(data_set)) ,  len(data_set) - int(0.8 * len(data_set)) ])
        train_dl = data.DataLoader(train_dataset, shuffle=True, batch_size=opt.batch_size, drop_last=False)
        valid_dl = data.DataLoader(valid_dataset, shuffle=True, batch_size=opt.batch_size, drop_last=False)
        optimizer = torch.optim.Adam(model_cdf.parameters(), lr=opt.learning_rate, weight_decay=opt.learning_rate_decay)

        best_micro = None
        loss_func = AuxLoss(top_k=2, device = device)

        for epoch in range(6):
            model_cdf.train()
            total_loss = 0
            for batch in train_dl:
                x , seq_len, y = batch

                optimizer.zero_grad()
                input_x = x.to(device)
                seq_len = seq_len.to(device)

                y = y.to(device)
                output = model_cdf(input_x, seq_len)
                loss = loss_func(output, y)
                # backward and optimize
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

            print('train loss: ', total_loss / len(train_dl))
            with torch.no_grad():
                for i, batch in enumerate(valid_dl):
                    x , seq_len, y = batch
                    input_x = x.to(device)
                    seq_len = seq_len.to(device)
                    output = model_cdf(input_x, seq_len)
                    v, idx = torch.topk(output, k=opt.top_k)
                    if i == 0:
                        y_pred = idx.to('cpu').numpy()
                        y_true = y
                    else:
                        y_pred = np.concatenate((y_pred, idx.to('cpu').numpy()))
                        y_true = np.concatenate((y_true, y))
            y_pre , y_true = idx2binary(y_pred, opt) , idx2binary(y_true, opt)
            metrics = evaluate(y=y_true, y_pre=y_pre)
            print('valid: ' , metrics)
            if (best_micro is None) or (best_micro <= metrics['instance_f1']):
                best_micro = metrics['instance_f1']
            else:
                break

        test_dataset = get_dataset_For_NN_Cdf(df=test_df, model=model, tokenizer=tokenizer, opt=opt, Seq_Len=48,
                                          device=device)
        test_dl = data.DataLoader(test_dataset, shuffle=True, batch_size=opt.batch_size, drop_last=False)
        with torch.no_grad():
            for i, batch in enumerate(test_dl):
                x, seq_len, y = batch

                input_x = x.to(device)
                seq_len = seq_len.to(device)
                output = model_cdf(input_x, seq_len)
                v, idx = torch.topk(output, k=opt.top_k)
                if i == 0:
                    y_pred = idx.to('cpu').numpy()
                    y_true = y
                else:
                    y_pred = np.concatenate((y_pred, idx.to('cpu').numpy()))
                    y_true = np.concatenate((y_true, y))
            y_pre, y_true = idx2binary(y_pred, opt), idx2binary(y_true, opt)
            metrics = evaluate(y=y_true, y_pre=y_pre)
            print('Test 散装评论大集合: NN_cdf' , metrics)
    else:
        df = test_df
        top_k = 3 # 只取 y_pred 的最大值 TODO: 其它设置
        with torch.no_grad():
            for i in range(len(df)):
                file_idx = df['abstract_df_idx'].iloc[i]
                hashtag =  df['hashtag'].iloc[i]
                Comments, _ = get_SingleLabelFromCIDC_rawLabel(file_idx)
                Comments = [ hashtag + '[SEP]' + c  for c in Comments ]
                single_dataset = SingleML_DataSet(Comments, tokenizer, opt)
                tmp_dl = data.DataLoader(single_dataset , batch_size=opt.batch_size, drop_last=False, shuffle=True)

                for j, batch in enumerate(tmp_dl):
                    x, mask = batch
                    input_x = x.to(device)
                    mask = mask.to(device)
                    output = model(input_x, mask)
                    if j == 0:
                        y_1 = output.softmax(dim=-1).sum(dim=0)
                        y_2 = torch.zeros(opt.label_num).to(device)
                        for idx in torch.topk(output, dim=1, k=top_k)[1]:  # 后续可 topK
                            y_2[idx] += 1
                    else:
                        y_1 += output.softmax(dim=-1).sum(dim=0)
                        for idx in torch.topk(output, dim=1, k=top_k)[1]:  # 后续可 topK
                            y_2[idx] += 1
                # TODO 开始你的表演： 统计学硕士同志 QWQ
                all_idx = torch.topk(y_2 , k=opt.top_k)[1] # 计数法
                all_idx_1 = torch.topk(y_1, k=opt.top_k)[1]
                if i == 0:
                    y_pred = all_idx.to('cpu').unsqueeze(0).numpy()
                    y_pred_1 = all_idx_1.to('cpu').unsqueeze(0).numpy()
                else:
                    y_pred = np.concatenate((y_pred , all_idx.to('cpu').unsqueeze(0).numpy() ) )
                    y_pred_1 = np.concatenate((y_pred_1, all_idx_1.to('cpu').unsqueeze(0).numpy()))

        y_true = np.array(list(df['raw_labels'].apply(lambda x: eval(x))))
        y_pred ,y_true = idx2binary(y_pred, opt) , idx2binary(y_true, opt)
        y_pred_1 = idx2binary(y_pred_1 , opt)
        metrics = evaluate(y=y_true, y_pre=y_pred)
        print('逐个计数法： ', top_k , metrics)
        metrics = evaluate(y=y_true, y_pre=y_pred_1)
        print('连续相加法： ',metrics)
    return


def create_data(File_Paths):
    df = pd.read_csv( File_Paths['df'], encoding='utf-8')
    train_df,valid_df,test_df,a,b,c = split_train_test(df,  df.copy())
    df_origin = train_df

    save_path = '/root/MLabelCls/data/Hashtag_emoji/Comments_CIDC_trainLM.csv'
    record_data = []
    for i in range(len(df_origin)):
        file_idx = df_origin['abstract_df_idx'].iloc[i]
        raw_labels = eval(df_origin['raw_labels'].iloc[i])
        if len(raw_labels) < 3:
            assert False
        tmp_txt = df_origin['hashtag'].iloc[i]
        Prior = { (raw_labels[0],):0,(raw_labels[1],):1 ,(raw_labels[2],):2,} # 相关性强的评论来训语言模型
        Prior_comments = get_CommentFromCIDC_rawLabel(file_idx, Prior, Maxi_num=5)
        for key , comments in enumerate(Prior_comments):
            label_idx = raw_labels[key]
            for c in comments:
                record_data.append( ( tmp_txt + '[SEP]' + c , label_idx ) )
    new_df = pd.DataFrame(record_data , columns=['corpus','label'])
    new_df.to_csv(save_path, index=False)
    return



def testTransformer():
    a = transformers.EncoderDecoderModel()


