import torch
from transformers import BertTokenizer
import random

DEFAULT_MODEL_NAME = '/root/Bert_preTrain'


class Block:
    tokenizer = BertTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
    def __init__(self, ids, pos, blk_type=1, relevance = 0 ,**kwargs):
        self.ids = ids
        self.pos = pos
        self.blk_type = blk_type  # 0 query or hashtag, 1 普通评论 # ?? 直白视作原始标签21
        self.relevance_nature = relevance # 先天判定的 relevance
        self.relevance_acquired = 0 # 后天的 relevance
        self.estimation = 0
        self.__dict__.update(kwargs)

    # 对于本任务, position 不存在因果上的先后, 排不排序无所谓 // 或许，可以让estimation大的放前面
    def __lt__(self, rhs):
        return self.blk_type < rhs.blk_type or (self.blk_type == rhs.blk_type and self.pos < rhs.pos)

    def __ne__(self, rhs):
        return self.pos != rhs.pos or self.blk_type != rhs.blk_type

    def __len__(self):
        return len(self.ids)

    def __str__(self):
        return Block.tokenizer.convert_tokens_to_string(Block.tokenizer.convert_ids_to_tokens(self.ids))


class Buffer:
    def __init__(self , block_list = [] , hashtag_head = True):
        self.blocks = block_list
        self.hashtag_head = hashtag_head

    def __add__(self, buf):
        ret = Buffer()
        ret.blocks = self.blocks + buf.blocks
        return ret

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, key):
        return self.blocks[key]

    def __str__(self):
        return ''.join([str(b) + '\n' for b in self.blocks])

    def clone(self):
        ret = Buffer()
        ret.blocks = self.blocks.copy()
        return ret

    def calc_size(self):
        return sum([len(b) for b in self.blocks])

    def calc_relevance_size(self):
        return sum([len(b) for b in self.blocks if (b.relevance_nature + b.relevance_acquired >= 1) ])

    def block_ends(self):
        t, ret = 0, []
        for b in self.blocks:
            t += len(b)
            ret.append(t)
        return ret

    def insert(self, b, reverse=True):
        if not reverse:
            for index in range(len(self.blocks) + 1):
                if index >= len(self.blocks) or b < self.blocks[index]:
                    self.blocks.insert(index, b)
                    break
        else:
            for index in range(len(self.blocks), -1, -1):
                if index == 0 or self.blocks[index - 1] < b:
                    self.blocks.insert(index, b)
                    break

    def merge(self, buf):
        ret = Buffer()
        t1, t2 = 0, 0
        while t1 < len(self.blocks) or t2 < len(buf):
            if t1 < len(self.blocks) and (t2 >= len(buf) or self.blocks[t1] < buf.blocks[t2]):
                ret.blocks.append(self.blocks[t1])
                t1 += 1
            else:
                ret.blocks.append(buf.blocks[t2])
                t2 += 1
        return ret

    def filtered(self, fltr: 'function blk, index->bool', need_residue=False): # ??
        ret, ret2 = Buffer(), Buffer()
        for i, blk in enumerate(self.blocks):
            if fltr(blk, i):
                ret.blocks.append(blk)
            else:
                ret2.blocks.append(blk)
        if need_residue:
            return ret, ret2
        else:
            return ret

    def random_sample(self, size):
        assert size <= len(self.blocks)
        index = sorted(random.sample(range(len(self.blocks)), size))
        ret = Buffer()
        ret.blocks = [self.blocks[i] for i in index]
        return ret

    def sort_(self):
        self.blocks.sort()
        return self

    def sort_by_estimation(self):
        self.blocks.sort(key=lambda x: -(x.estimation + 0.1*x.relevance_nature + 0.1*x.relevance_acquired) ) # 从大到小排列
        return

    def fill(self, buf):
        ret, tmp_buf, tmp_size = [], self.clone(), self.calc_size()
        for blk in buf:
            if tmp_size + len(blk) > 10: # CAPACITY
                ret.append(tmp_buf)
                tmp_buf, tmp_size = self.clone(), self.calc_size()
            tmp_buf.blocks.append(blk)
            tmp_size += len(blk)
        ret.append(tmp_buf)
        return ret


    def export(self,drop_idx_set, total_length = None , drop_cls=True  ):
        '''
            目标只有一个: 将 buffer 整成reasoner的输入
        '''
        if total_length is None:
            total_length = self.calc_relevance_size() // 3
        # 初始化
        ids, att_masks = torch.zeros(2, total_length, dtype=torch.long) # dtype=torch.long
        t = 0
        for i , b in enumerate(self.blocks):
            if i in drop_idx_set: continue
            elif t + len(b) > total_length: break

            if drop_cls and i > 0:
                ids[t:t + len(b)-1] = torch.tensor(b.ids[1:], dtype=torch.long)  # id
                att_masks[t:t + len(b)-1 ] = 1  # attention_mask
                t += len(b)-1
            else:
                ids[t:t + len(b)] = torch.tensor(b.ids, dtype=torch.long,)  # id
                att_masks[t:t + len(b)] = 1  # attention_mask
                t += len(b)
        # 注 此时ids:  [x x x x [sep] 0 0 0]
        return ids, att_masks


    def export_blocks(self, with_head = True):
        if with_head and self.hashtag_head:
            head_id = self.blocks[0].ids
            Blk_ids , Relevance_natrue, Relevance_acquired = [] , [] , []
            for blk_idx in range(1, len(self)):
                blk = self.blocks[blk_idx]
                Blk_ids.append(head_id + blk.ids )
                Relevance_natrue.append( blk.relevance_nature  )
                Relevance_acquired.append(blk.relevance_acquired)
            return Blk_ids , Relevance_natrue, Relevance_acquired
        else:
            Blk_ids , Relevance_natrue, Relevance_acquired = [] , [] , []
            for blk_idx in range(1, len(self)):
                blk = self.blocks[blk_idx]
                Blk_ids.append( blk.ids )
                Relevance_natrue.append( blk.relevance_nature  )
                Relevance_acquired.append(blk.relevance_acquired)
            return Blk_ids , Relevance_natrue, Relevance_acquired


    def extract_Z_buffer(self, Maxi_len = None):
        if Maxi_len is None: Maxi_len = self.calc_relevance_size() // 2
        # a.relevance by estimation + b. 随机抽样
        self.blocks[0].estimation = 1 # 足够大的数
        self.blocks[0].relevance_acquired = 1
        self.sort_by_estimation()
        # 直接 0.25 relevant // 0.05 随机
        Z_blk_list = []
        total_length = 0
        for i in range( len(self) // 2 ): # TODO 至多前1/2条
            total_length += len(self.blocks[i])
            if total_length > Maxi_len: break
            Z_blk_list.append(self.blocks[i])
        return Buffer(Z_blk_list , hashtag_head=self.hashtag_head)


    def export_relevance(self, device, length=None, dtype=torch.long, out=None):
        if out is None:
            total_length = self.calc_size() if length is None else length * len(self.blocks)
            relevance = torch.zeros(total_length, dtype=dtype, device=device)
        else:
            relevance = out
        t = 0
        for b in self.blocks:
            w = t + (len(b) if length is None else length)
            if b.relevance >= 1:
                relevance[t: w] = 1
            t = w
        return relevance
