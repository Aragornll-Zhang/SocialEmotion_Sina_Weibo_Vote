import re
# import bs4.element
import requests
from bs4 import BeautifulSoup
import pandas as pd
import bs4
import time
import random
import heapq

# TODO CIC 再想更进一步爬全，就得想着搞它一波动态页面
# TODO: 假装尊重一波微博 time.sleep(random.uniform(2,3))


# 需Weibo端 clear Cookie 发挥作用, 详见 https://www.kancloud.cn/jyunwaa/weibo/1881264
Cookie = "Your cookie" # 微博账号，用户cookie
user_agent = "Your request head(可以为 空 )" # request 请求头
headers = {'Cookie':Cookie ,'User-Agent':user_agent }


# * --------------------  NEW THING After 3_15  --------------- *
def filter_func(comment, method = 0):
        '''
            通用讨论区讨论评论 + 针对某条评论的 "评中评"
            ps 核心筛选理由，根据 <img , alt = '[Doge]'  xxx> 这样的Tag
        :param comment: soup.Tag 类型
        :param method:
        :return:
        '''
        if method == 0:
            if comment.find(lambda tag: (tag.name == 'img') and ('alt' in tag.attrs)): # 有个表情就行
                return True
            else: return False
        elif method == 1:
            pass



def get_comments_In_one_Page(comment_page_url , Filter = True , Heap_mid = []):
    '''
        爬讨论区，某一页中的所有评论内容
        注
    :param comment_page_url: e.g. https://s.weibo.com/weibo?q=%2305后公布恋情的方式%23&page=3
    :param filter: Bool， True: 启动筛选机制
    :return:
    '''
    # comment_page_url = "https://s.weibo.com/weibo?q=%2305后公布恋情的方式%23&page=5"
    global HEAP_MAXI_LEN
    # comment_page_url = "https://s.weibo.com/weibo?q=%23%E5%AD%9D%E6%84%9F%E5%B0%8F%E5%8C%BA%E8%BF%91%E7%99%BE%E4%BA%BA%E8%81%9A%E9%9B%86%E7%85%BD%E5%8A%A8%E8%80%85%E8%A2%AB%E6%8D%95%23&page=1"
    # *----------- 经典模板 BEGIN -------------------*
    resp = requests.get(comment_page_url, headers= headers , timeout=(2, 4))
    if resp.status_code != 200: assert False # TODO Try Except
    resp.encoding = 'utf-8'  # resp.apparent_encoding
    soup = BeautifulSoup(resp.text , features="lxml") # or features='html.parser'
    # *----------- 经典模板 END -------------------*
    # find 属于模糊搜索  select 为精确法

    # 要获得微博评论的顺序文本, ##，所以要遍历一波
    # e.g. 某种带hashtag / 表情的顺序文本: 05后公布恋情的方式#我是07年的，还没有遇到喜欢我的男生，我可能是个假的05后。[允悲][允悲][允悲]

    # ps 为消去某不优异空格符 '/u200b'
    DELETE = "\u200b"
    All_page_comments_record = []

    # 爬评论数目
    for comment_i in soup.findAll(lambda tag: tag.name == 'div' and ( 'mid' in tag.attrs)  ):
        # Step 1. 爬评论数 + mid
        try:
            cur_mid = str( comment_i['mid'] )
            tmp_str = comment_i.select_one('div[class=card-act]').select('li')[1].text
            tmp_str = re.sub("[^0-9]",'', tmp_str ) # 去掉非数字字符
            if len(tmp_str) > 0:
                CIC_num = int(tmp_str)
                if len(Heap_mid) < HEAP_MAXI_LEN:
                    heapq.heappush(Heap_mid , (CIC_num , cur_mid))
                else:
                    heapq.heappushpop(Heap_mid, (CIC_num, cur_mid))
        except:
            print('wrong in url: {} , comment_i: {}'.format(comment_page_url , comment_i['mid'] ) )

        # Step 2. 爬评论
        comment_i_content = comment_i.find('p', {'node-type': "feed_list_content_full", 'class': 'txt'}) # or ("p", style="display: none")
        if not comment_i_content:
            # 无 "展开" 页
            comment_i_content = comment_i.find('p', {'node-type': "feed_list_content", 'class': 'txt'})
        if Filter: # 玩微博呐，要整点表情
            # TODO: 后期再筛选需要的表情 / 做一个词映射
            # e.g.
            if not filter_func( comment_i_content, method=0 ): continue

        raw_text = ""
        for tag in comment_i_content.children: # 顺序遍历所有儿子节点
            try:
                if isinstance(tag , bs4.element.Tag):
                    if tag.name == 'img' and ('alt' in tag.attrs):
                        raw_text += str( tag['alt'] ) # emoji
                    else: # 其它 tag
                        raw_text += tag.text.strip('\n')
                elif isinstance(tag , bs4.element.NavigableString):
                    raw_text += re.sub( DELETE , '' , tag.text.strip(' ').strip('\n') )
            except:
                continue
        All_page_comments_record.append(raw_text)

    return All_page_comments_record



def crawl_CommentsInDiscussion(url, index ):
    '''
    :param url:  热搜hashtag讨论区 url, e.g. https://s.weibo.com/weibo?q=%2305后公布恋情的方式%23
    :param index: 确定 “save_path”具体名
    :return:
    '''
    # https://s.weibo.com/weibo?q=%2305后公布恋情的方式%23&page=1
    # 有的小哥们自带表情!!!，我们可以假设...

    # url = 'https://s.weibo.com/weibo?q=%2305后公布恋情的方式%23' # nodup广告太多

    # Method 1: 讨论区无差别留言  url 例子: https://s.weibo.com/weibo?q=%2305后公布恋情的方式%23

    surffix_0 = '&nodup=1' # 全爬模式 垃圾广告太多，还是得靠咱们的weibo筛选一波
    # *----------- 经典模板 BEGIN -------------------*
    # resp = requests.get(url + surffix_0, headers= headers , timeout=(2, 4))
    # if resp.status_code != 200:
    #     print('404 not found extra suffix')
    #     resp = requests.get(url, headers=headers, timeout=(2, 4))
    #     if resp.status_code != 200:
    #         assert False # TODO Try Except
    # else:
    #     url += surffix_0


    resp = requests.get(url, headers= headers , timeout=(2, 4))
    if resp.status_code != 200:
        assert False # TODO Try Except
    resp.encoding = 'utf-8'  # resp.apparent_encoding
    soup = BeautifulSoup(resp.text , features="lxml") # or features='html.parser'
    # *----------- 经典模板 END -------------------*

    CID_record = []
    # 获得评论区总页码数
    page_num = None
    try:
        page_num = len( soup.find('span' , {'class':'list'}).findAll('li') )
    except:
        print('can not get page num, url: {}'.format(url) )
        page_num = 1 # 爬不到，就整一页呗

    Heap_mid = [] # 一个记录 mid 的堆
    url_surffix = '&page={}'
    if page_num:
        for page in range(1, page_num+1):
            new_page_url = url + url_surffix.format(page)
            try:
                contents = get_comments_In_one_Page( new_page_url , Heap_mid = Heap_mid ) # 包含 emoji 的讨论区评论, type List[str]
                CID_record.extend(contents)
            except:
                continue
        # TODO 每一页停两秒
        time.sleep(random.uniform(2, 3))

    # for i in range(len(CID_record)):
    #     print(CID_record[i])
    # print(Heap_mid)
    # return

    # 写入文件
    # (i) comments
    save_path = '/Data_storage/CID/comment/{}.txt'.format(index)
    with open(save_path, 'a' , encoding='utf-8') as f:
        for i in range(len(CID_record)):
            f.write( CID_record[i] )
            f.write('\n')
    # (ii) 高评 mid
    save_mid_path = '/Data_storage/CID/mid/{}.txt'
    Heap_mid.sort(key=lambda x: -x[0])  # [(int , str) , (x,'xx') , ... ]
    write_heap = [ Heap_mid[k][1] for k in range(len(Heap_mid))]
    with open(save_mid_path.format(index), 'a', encoding='utf-8') as f:
        f.write(" ".join(write_heap))
        f.write('\n')
    return


def main_CID():
    '''
        CID: comments in discussion
        CIC: comments in comments
        自己动手，丰衣足食
    :return:
    '''
    # 只爬Abstract有的那些
    file_path = '/root/MLabelCls/data/Hashtag_emoji/HLA_39.csv'
    df = pd.read_csv( file_path, encoding='utf-8' ) # index from  0 --> len(df)-1

    # df = df.iloc[0:2]
    import os
    Past_Maxi = 0
    for f_name in os.listdir('/Data_storage/CID/comment/'):
        if f_name[-4:] == '.txt':
            Past_Maxi = max(Past_Maxi , int(f_name[0:-4]) )

    Hashtags = list(df['hashtag'])

    # CID , 每次截留评论数最多的N条评论， 为 CIC 做准备
    global HEAP_MAXI_LEN
    HEAP_MAXI_LEN = 10 # 取前10评论高的MID (反正多存点没坏处)


    hashtag_url = "https://s.weibo.com/weibo?q=%23{}%23" # 爬全乎
    for i in range( Past_Maxi+1 , len(df) ):
        try:
            crawl_CommentsInDiscussion(url=hashtag_url.format( Hashtags[i] ), index = i )
        except:
            print('wrong url here! {} , index: {}'.format( hashtag_url.format(Hashtags[i]) , i ) )
        time.sleep(1)
        if i % 1000 == 1:
            print('* ---------------------------  *')
            print('Already get here! {}'.format(i) )
            print('* ---------------------------  *')

    print('okk , we get all CID !!! ')
    return

# *--------- 爬最高评论 的 ‘论中论’-----------*
# 转换 mid(base 10) 到 url(base 62)
ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" # 62 进制表
def base62_encode(num, alphabet=ALPHABET):
    """Encode a number in Base X
    `num`: The number to encode
    `alphabet`: The alphabet to use for encoding
    """
    if (num == 0):
        return alphabet[0]
    arr = []
    base = len(alphabet)
    while num:
        rem = num % base
        num = num // base
        arr.append(alphabet[rem])
    arr.reverse()
    return ''.join(arr)


def mid_to_url(midint):
    # 转换规则是，url串值从最后往前，每四个字符为一组，作为一个62进制数，然后将各个62进制数转换成对应的10进制数，再将最终结果连接起来，就是该微博的id
    # 所以，如果知道"补零"与否,就能一一对应
    # mod 7 是因为‘ZZZZ’ = 62**4 - 1 = 14776335 , (weibo的mid的后7位必定小于此数)
    midint = str(midint)[::-1] # 倒序一波，走麻烦了兄弟
    size = len(midint) // 7 if len(midint) % 7 == 0 else len(midint) // 7 + 1
    result = []
    for i in range(size):
        s = midint[i * 7: (i + 1) * 7][::-1] # 等价于从后往前取
        s = base62_encode(int(s))
        s_len = len(s)
        if i < size - 1 and len(s) < 4: # 除了最后一波，都得补零(咱不知道微博的规则...)
            s = '0' * (4 - s_len) + s
        result.append(s)
    result.reverse()
    return ''.join(result)

# print(mid_to_url(4480317962180479) == 'IxHr699f1' )


Top50_emoji = {'[吃瓜]', '[失望]', '[摊手]', '[good]', '[嘻嘻]', '[馋嘴]', '[羞嗒嗒]', '[笑而不语]', '[加油]', '[赞]', '[思考]', '[疑问]', '[可爱]',
         '[并不简单]', '[哈哈]', '[挖鼻]', '[微笑]', '[鼓掌]',
         '[话筒]', '[污]', '[喵喵]', '[蜡烛]', '[悲伤]', '[拜拜]', '[汗]', '[可怜]', '[偷笑]', '[兔子]', '[伤心]', '[心]', '[怒]', '[太开心]',
         '[笑cry]', '[爱你]', '[费解]', '[中国赞]',
         '[酸]', '[给你小心心]', '[拳头]', '[允悲]', '[鲜花]', '[抱抱]', '[好喜欢]', '[二哈]', '[憧憬]', '[跪了]', '[泪]', '[微风]', '[doge]',
         '[作揖]'}

def get_CommentsInComent(Mid_url:str , index:int ,Filter:bool = True):
    '''
    :param Mid_url:  e.g. 62进制数 ‘IxHr699f1’ message ID url版
    :  return:
    '''

    # https://weibo.cn/comment/hot/IxHr699f1?&page=1  这哥们(IxHr699f1)是Messege Id(62进制数)    而一串数字的这个东西( 2028440564 )是 userId
    # 网页中爬到的mid(message ID 是十进制数)

    # Mid_url = 'IxHr699f1'
    # CIC_url = "https://weibo.cn/comment/hot/{}?&page={}".format(Mid_url , 1 )

    THRESHOLD = 200 # 最多爬x条带表情评论
    # CIC_url = "https://weibo.cn/comment/hot/IxHr699f1?&page=1" # 这哥们目前已经爬不全了...
    # # 这一块没必要了，反正也爬不全
    # # *----------- 经典模板 BEGIN -------------------*
    # resp = requests.get(CIC_url , headers= headers , timeout=(2, 4))
    # if resp.status_code != 200: assert False # TODO Try Except
    # resp.encoding = 'utf-8'  # resp.apparent_encoding
    # soup = BeautifulSoup(resp.text , features="lxml") # or features='html.parser'
    # # *----------- 经典模板 END -------------------*
    # try:
    #     total_page = int( soup.select_one('input[name=mp]')['value'] )
    # except:
    #     print('wrong in CIC page_num here')
    #     total_page = 1
    # total_page = min(total_page , 50)
    total_page = 50
    Record_commets = []

    last_Id = None # 每页首个Id
    for page_i in range(1, total_page+1):
        url = "https://weibo.cn/comment/hot/{}?&page={}".format(Mid_url, page_i)
        try:
            _resp = requests.get(url, headers=headers, timeout=(2, 4))
            if _resp.status_code != 200: assert False  # TODO Try Except
            _resp.encoding = 'utf-8'  # resp.apparent_encoding
            _soup = BeautifulSoup(_resp.text, features="lxml")  # or features='html.parser'

            cur_id = _soup.find(lambda tag: (tag.name == 'div') and ('id' in tag.attrs) )['id']
            if last_Id and last_Id == cur_id:
                print('last Page {} , Id {}'.format(page_i , last_Id))
                break # 别爬了，该网页的潜力挖到头了 TODO 怎么想着再搞他一波
            else:
                last_Id = cur_id

            DELETE = re.compile('评论配图|回复@|回复|收起d')
            # 爬单页评论,整一个 break 机制(假如上页跟此页相同，break， weibo反爬了2333)

            for comment_i in _soup.select('span[class=ctt]'):
                if Filter:  # 玩微博呐，要整点表情
                    # 直接后续再整 TODO: 后期再筛选需要的表情 / 做一个词映射
                    if not filter_func(comment_i , method=0): continue
                raw_text = ""
                Valid_Emoji_FLAG = False # in Top50
                for tag in comment_i.children:  # 顺序遍历所有儿子节点
                    # print(type(tag))
                    if isinstance(tag, bs4.element.Tag):
                        if tag.find( 'img' and ('alt' in tag.attrs)):
                            cur_emoji = str(tag.find('img' and ('alt' in tag.attrs))['alt'])
                            raw_text += cur_emoji
                            if cur_emoji in Top50_emoji: # 筛选Top50
                                Valid_Emoji_FLAG = True
                        elif tag.name == 'img' and ('alt' in tag.attrs):
                            raw_text += str(tag['alt'])  # emoji
                            if str(tag['alt']) in Top50_emoji:
                                Valid_Emoji_FLAG = True
                        else:  # 其它 tag
                            raw_text += tag.text.strip('\n')
                    elif isinstance(tag, bs4.element.NavigableString):
                        raw_text += re.sub(DELETE, '', tag.text.strip(' ').strip('\n'))
                if Valid_Emoji_FLAG: # 只筛选有用的 emoji
                    Record_commets.append( re.sub(DELETE, '', raw_text.strip(' ').strip('\n')) )
                if len(Record_commets) >= THRESHOLD:
                    return Record_commets

        except:
            print('wrong someplace')
            time.sleep(random.uniform(1, 2))

        # 每爬一页, 歇0.5秒
        time.sleep(random.uniform(0, 0.25))

    # 写入文件
    # comments in comments
    save_path = 'Your save_path/{}.txt'.format(index)
    with open(save_path, 'a' , encoding='utf-8') as f:
        for i in range(len(Record_commets)):
            f.write( Record_commets[i] )
            f.write('\n')
    return



def main_CIC():
    # 只爬Abstract有的那些
    file_path = '/root/MLabelCls/data/Hashtag_emoji/HLA_39.csv'
    df = pd.read_csv( file_path, encoding='utf-8' ) # index from  0 --> len(df)-1

    # df = df.iloc[0:1000]
    import time
    start = time.time()

    import os
    Past_Maxi = -1
    for f_name in os.listdir('/Data_storage/CIC/comment/'):
        if f_name[-4:] == '.txt':
            Past_Maxi = max(Past_Maxi , int(f_name[0:-4]) )

    All_mid =  set( os.listdir('/Data_storage/CID/mid/') ) # 这个Index是按 HLA_39 的顺序编码的

    global MAXI_MID_USED_NUM
    MAXI_MID_USED_NUM = 4 # 前四条高评

    # with open('/Data_storage/CID/mid/' + str(4743) + '.txt', 'r', encoding='utf-8') as f:
    #     line = f.read().strip('\n')

    for i in range(Past_Maxi + 1, len(df)):
        idx = i
        # Mid_index = Index_s[i] # For file search
        if str(i) + '.txt' in All_mid:
            with open('/Data_storage/CID/mid/' +  str(i) + '.txt' , 'r' , encoding='utf-8' ) as f:
                l = f.read().strip('\n')
            try:
                line = l.split(' ')
                for i in range(min(MAXI_MID_USED_NUM , len(line))):
                    mid_base10 = line[i]
                    Mid = mid_to_url(mid_base10) # e.g. IE0GfiQXp   type(Mid) = str
                    get_CommentsInComent(Mid_url=Mid , index=idx)
            except:
                print('sth. wrong!!! index:{} '.format(idx))
            time.sleep(1)
        if i % 100 == 1:
            print('* ---------------------------  *')
            print('Already get here! {}'.format(i))
            print('Time: {} min'.format( (time.time() - start) // 60 ) )
            print('* ---------------------------  *')
        break

    print( 'Train End. Already {} hashtags in {} minutes'.format( len(df) - Past_Maxi - 1  , (time.time() - start) // 60 ) )
    print('okk , we get all CIC !!! ')
    return



if __name__ == '__main__':
    print('okk')
    # main()
    # Crawl_All_emotions()
    # main_CID()
    # main_CIC()