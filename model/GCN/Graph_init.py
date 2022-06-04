import numpy as np

EMOTION_THRESHOLD = 0.0 # e.g. =0.1 means 至少10%, 才列为 target label, 此时 1/2/3 = 16/66/18
Gamma_Threshold = 0.1  # 这个标签也忒不经用了！！ 太分散了，0.4 时，一个都没有……
p_reserve = 0.2 # 计算概率矩阵时,给对角元素的权重
emoji2idx = {'[心]': 0, '[费解]': 1, '[doge]': 2, '[允悲]': 3, '[泪]': 4, '[赞]': 5, '[怒]': 6, '[憧憬]': 7,
             '[笑cry]': 8, '[吃瓜]': 9, '[蜡烛]': 10, '[微笑]': 11, '[伤心]': 12, '[给力]': 13, '[中国赞]': 14, '[吃惊]': 15,
             '[跪了]': 16, '[给你小心心]': 17, '[嘻嘻]': 18, '[拜拜]': 19, '[酸]': 20, '[并不简单]': 21, '[思考]': 22, '[威武]': 23}

idx2emoji = {0: '心', 1: '费解', 2: 'doge', 3: '允悲', 4: '泪', 5: '赞', 6: '怒', 7: '憧憬',
             8: '笑cry', 9: '吃瓜', 10: '蜡烛', 11: '微笑', 12: '伤心', 13: '给力', 14: '中国赞', 15: '吃惊',
             16: '跪了', 17: '给你小心心', 18: '嘻嘻', 19: '拜拜', 20: '酸', 21: '并不简单', 22: '思考', 23: '威武'}


def count_Co_Occurence(raw_labels , dim = 24 ):
    '''
    :param raw_labels: 多标签分类的 ground truth, e.g.
           [[label a, label b] , [label a] , [label b , label c , label a] , ...]
    :param dim: 标签总数
    :return:
    '''
    Mat = [[0] * dim for _ in range(dim)]  # symmetric
    for labels in raw_labels:
        for i in range(len(labels)):
            for j in range(i, len(labels)):
                x = labels[i]
                y = labels[j]
                Mat[x][y] += 1
                if x != y:
                    Mat[y][x] += 1 # 之前的不对称...
    return Mat


def get_Corr_Mat(Mat , clip_diag = True):
    '''
    :param Mat:
    :param clip_diag: 去除 Mat[i,i] 算阈值
    :return:
    '''
    N = len(Mat)
    Mat = np.array(Mat , dtype=float)
    if not all(Mat.sum(axis=0) ):
        print('a label occurs less than once! ')
        assert False
    # P_ij = P( L_j | L_i ) = Mat[i,j] / Mat[i,:]
    # P = Mat / Mat.sum(axis=1)
    P =  np.zeros_like(Mat)
    for i in range(N):
        all_sum = max( np.sum(Mat[i,:]) - Mat[i,i]  , 1.0 )
        for j in range(N):
            if Mat[i,j] / all_sum  >= Gamma_Threshold:
                P[i][j] = 1.0
            else:
                P[i][j] = 0.0

    # re-weight to avoid over-smoothing (防止标签embedding太平均)
    A = np.zeros_like(P) # A_ij = p / ( A[i,:] - A[i,i] ) , re-weighted
    for i in range(N):
        col_sum = np.sum( P[i,:] )
        for j in range(N):
            if P[i,j] > 0:
                A[i,j] =  p_reserve / (col_sum-1)  if i != j else 1-p_reserve
    return A


def get_target(file_name = "/root/MLabelCls/data/Hashtag_emoji/hashtag_emotion.txt", offset = 0):
    record = []

    with open(file_name , 'r' , encoding='utf-8') as f:
        for line in f.readlines():
            try:
                tmp = []
                line = line.strip('\n').split('\t')
                total_vote = int(line[2])
                top1 , top1_num = line[3].split(':')[0] , int( line[3].split(':')[1] )
                tmp.append(emoji2idx[top1]+offset)
                if len(line) >= 5:
                    top2, top2_num = ( line[4].split(':')[0], int(line[4].split(':')[1]) )
                    if top2_num / total_vote >= EMOTION_THRESHOLD:
                        tmp.append(emoji2idx[top2] + offset)
                if len(line) >= 6:
                    top3, top3_num = ( line[5].split(':')[0], int(line[5].split(':')[1]) )
                    if top3_num / total_vote >= EMOTION_THRESHOLD:
                        tmp.append(emoji2idx[top3]+offset)
                record.append(tmp)
            except:
                print( line[0] ," wrong here!" )
                assert False
    return record

# record = get_target()

# #################### *--- 用到的邻接矩阵生成函数  ---* #############
def get_adjMat(raw_target, opt):
    '''
        按照 GCN 原文生成 adj
    :return:
    '''
    Mat = count_Co_Occurence(raw_target , dim = 24)
    Mat = np.array(Mat, dtype=float)
    gamma_Threshold = opt.gamma_Threshold # 可调节
    if not all(Mat.sum(axis=0) ):
        print('a label occurs less than once! ')
        assert False
    import scipy.sparse as sp
    N = Mat.shape[0]
    P =  np.zeros_like(Mat)
    for i in range(N):
        all_sum = max( np.sum(Mat[i,:]) - Mat[i,i]  , 1.0 )
        for j in range(N):
            if Mat[i,j] / all_sum  >= gamma_Threshold:
                P[i][j] = 1.0
            else:
                P[i][j] = 0.0

    def normalize_adj(adj):
        '''
            Symmetrically normalize adjacency matrix.
                A_tilda = D^-1/2 * A * D^-1/2
            # 归一化
        :param adj:
        :return:
        '''
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    adj = normalize_adj(P) # step1: A = D-0.5 * A * D-0.5
    A_tilda = normalize_adj( adj + sp.eye(N) ).todense() # step2: return numpy matrix
    return A_tilda


# 边角料, 生成数据
def generate_TargetLabels():
    # one-hot 编码
    record = get_target()
    target = []
    for labels in record:
        cur = [0] * len(emoji2idx) # 24
        for i in range(len(labels)):
            cur[labels[i]] = 1
        target.append( tuple(cur) )
    import pandas as pd
    df = pd.read_csv("/root/MLabelCls/data/Hashtag_emoji/HashtagAndLabels.csv" , usecols=['hashtag'] )
    assert len(target) == len(df) == len(record)
    df.insert( loc=1 , column= 'one_hot' , value= target )
    df.to_csv('/root/MLabelCls/data/Hashtag_emoji/HashtagAndLabels_00.csv' , index=False )
    return
    # add raw labels
    # record = get_target()
    # path_name = "/root/MLabelCls/data/Hashtag_emoji/HashtagAndLabels_10.csv"
    # df = pd.read_csv(path_name)
    # df.insert(loc=1 , column= 'raw_labels' , value= record)
    # df.to_csv( path_name , index=False )


def SGM_rawTargetlabels():
    import pandas as pd
    extra = 3 # pad, bos , eos
    record = get_target(offset = extra)
    path_name = "/root/MLabelCls/data/Hashtag_emoji/HashtagAndLabels_00.csv"
    save_path = "/root/MLabelCls/data/Hashtag_emoji/HashtagAndLabels_SGM_00.csv"
    df = pd.read_csv(path_name)
    df.insert(loc=1 , column= 'raw_labels' , value= record)
    df.to_csv( save_path , index=False )
    return




def main():
    file_name = "F:\specific_python\FinalGame\data\Hashtag_emoji\hashtag&emotion.txt"
    target = get_target(file_name)
    # print(target)
    Mat = count_Co_Occurence(target , dim = 24)
    A_reweight = get_Corr_Mat(Mat)
    save_path = "A_rewight_gamma_{}p_{}.npy".format(int(Gamma_Threshold*100) , int(p_reserve*10))
    np.save(save_path , A_reweight)
    print('save... ok')
    print("### only for test ###")
    arr = np.load(save_path)
    print(arr.shape , arr)
    return


if __name__ == '__main__':
    print('okk, 3/1')
    # main()
    # generate_TargetLabels()
