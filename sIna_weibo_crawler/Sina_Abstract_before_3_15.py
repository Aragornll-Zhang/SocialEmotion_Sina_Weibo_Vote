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
# TODO:  time.sleep(random.uniform(2,3))


# 需Weibo端 clear Cookie 发挥作用, 详见 https://www.kancloud.cn/jyunwaa/weibo/1881264
Cookie = "Your cookie" # 微博账号，用户cookie
user_agent = "Your request head(可以为 空 )" # request 请求头
headers = {'Cookie':Cookie ,'User-Agent':user_agent }
# *-------- BEFORE 3-15 ----------*


# 此站点可爬所有标签信息
# https://m.s.weibo.com/ajax_hot/attitudeList?query=高晓松涂口红

# 群体表情标签:
# xxx = hashtag, e.g. #高晓松涂口红#
# https://m.s.weibo.com/hot/attitude?query=xxx e.g. https://m.s.weibo.com/hot/attitude?query=%23高晓松涂口红%23
# https://github.com/keyucui/weibo_topic_analyze 爬微博, 评论分析

def get_abstract(hashtag_url):
    '''
       仅简单地获取导言
    :param hashtag_url:
    :return:
    '''
    # e.g. hashtag_url = "https://s.weibo.com/weibo?q=%23谷爱凌的新挑战%23"
    # '导语：敢于突破，才能抵达更高段位，DW持妆粉底液，采用突破性锁妆校色科技，上妆即锁色，24H持妆透气不暗沉，助天才少女谷爱凌锁定无瑕状态，轻妆上阵，赢得漂亮！'
    # hashtag_url = "https://s.weibo.com/weibo?q=%2305后公布恋情的方式%23"
    resp = requests.get(hashtag_url, headers= headers , timeout=(2, 4))
    if resp.status_code != 200: assert False # TODO Try Except
    resp.encoding = 'utf-8'  # resp.apparent_encoding
    soup = BeautifulSoup(resp.text , features="lxml") # or features='html.parser'
    # soup.prettify()
    introduction = None
    for child in soup.findAll('div',{"class": "card card-topic-lead s-pg16"}): # 其中一种方式
        if child.find('p'):
            introduction = child.find('p').text
            break
    assert introduction
    return introduction


def get_single_emojis(url):
    # 获得标签的所有emoji
    # url = "https://m.s.weibo.com/ajax_hot/attitudeList?query=高晓松涂口红"
    resp = requests.get(url, headers= headers , timeout=(2, 4))
    if resp.status_code != 200: assert False # TODO Try Except
    try:
        json = resp.json()
        info_list = json['data']['total']
        emoji = [0] * len(info_list)
        cnt = [0] * len(info_list)
        for i in range(len(info_list)):
            dict_i = info_list[i] # e.g. {'emoji_id': 6, 'count': 364, 'url': '//img.t.sinajs.cn/t4/appstyle/searchpc/css/h5/claim/img/face/face6.png?v=20210924174800', 'id': 6}
            emoji[i]  = dict_i['emoji_id']
            cnt[i] = dict_i['count']
    except:
        print('Wrong here! url: {}'.format(url) )
    return emoji , cnt


def Crawl_All_emotions():
    '''
        爬全情感标签, 以备不时之需(主要为了假装有点事情干)
    # :param hashtag_url: https://m.s.weibo.com/ajax_hot/attitudeList?query=高晓松涂口红 这种格式
    :return:
    '''
    path = "/root/MLabelCls/data/Hashtag_emoji/HashtagAndLabels_00.csv"
    save_path = "/root/MLabelCls/data/Hashtag_emoji/HAndRawLabels.csv"

    df = pd.read_csv(path , encoding='utf-8')

    # df = df.iloc[0:10]

    Hashtags = list(df['hashtag'])
    hashtag_url = 'https://m.s.weibo.com/ajax_hot/attitudeList?query={}'
    All_raw_labels = []
    All_cnt = []
    fail_to_get = []


    for i in range(len(Hashtags)): # len(Hashtags)
        try:
            emoji_s , cnt = get_single_emojis(url=hashtag_url.format(Hashtags[i]))
        except:
            emoji_s = []
            cnt = []
            fail_to_get.append(i)
            print('All failed num here {},  url: {} '.format( len(fail_to_get), hashtag_url.format(Hashtags[i]) ) )
        All_raw_labels.append(emoji_s)
        All_cnt.append(cnt)
        if i % 1000 == 0:
            print('Already crawl {} attitude! '.format(i) )

    assert len(All_raw_labels) == len(All_raw_labels)

    data = [(i , Hashtags[i] , tuple(All_raw_labels[i]) , tuple(All_cnt[i] ) ) for i in range(len(Hashtags))]
    new_df = pd.DataFrame(data=data , columns=['idx' ,'hashtag' , 'emoji_s' , 'count'])
    new_df.to_csv(save_path , index=False)
    print(new_df.head())
    return


def main():
    path = "/root/MLabelCls/data/Hashtag_emoji/HashtagAndLabels_00.csv"
    save_path = "/root/MLabelCls/data/Hashtag_emoji/HashtagAndAbstract.csv"
    df = pd.read_csv(path , encoding='utf-8')
    Hashtags = list(df['hashtag'])

    fail_to_get = []
    hashtag_url = "https://s.weibo.com/weibo?q=%23{}%23"
    singleFile_name = "/root/MLabelCls/data/Hashtag_emoji/abstract/{}.txt"

    import os
    maxi = 0
    for item in os.listdir(path = "/root/MLabelCls/data/Hashtag_emoji/abstract" ):
        if int(item[0:-4]) > maxi:
            maxi = int(item[0:-4])

    Abstract = [''] * maxi # 之前有的
    for i in range(maxi, len(Hashtags)): # len(Hashtags)
        try:
            text = get_abstract(hashtag_url=hashtag_url.format(Hashtags[i]))
            with open(singleFile_name.format(i) , 'w' ,encoding='utf-8') as f:
                f.write(text)
        except:
            text = ''
            fail_to_get.append(i)
            print('failed here {},  url: {} '.format(i, hashtag_url.format(Hashtags[i]) ) )
        Abstract.append(text)
    print(len(fail_to_get))
    assert len(Abstract) == len(Hashtags)
    data = [(Hashtags[i] , Abstract[i]  ) for i in range(len(Hashtags))]
    new_df = pd.DataFrame(data=data , columns=['hashtag' , 'abstract'])
    new_df.to_csv(save_path , index=False)
    return