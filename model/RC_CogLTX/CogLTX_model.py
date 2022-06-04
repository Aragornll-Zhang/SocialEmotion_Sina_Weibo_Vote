import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertPreTrainedModel
# from RC_CogLTX.CogLTX_utils import Judge_Config


class Introspector(nn.Module):
    def __init__(self, opt  ):
        super(Introspector, self).__init__()
        self.bert = BertModel.from_pretrained( opt.PLM_name )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(opt.hidden_dim, 2)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        # sequence_output = outputs[0] # 整个sequence
        sequence_output = outputs[1] # [cls], 但我们只需关心整句分类

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output) # 为判断是否 relevant # TODO: CE loss

        outputs = logits # ? 这句是防 block 没标签
        return outputs  # (loss), scores, (hidden_states), (attentions)


class Reasoner(nn.Module): # Interface # 我直接当成bert分类器，跟上述一样
    def __init__(self, opt):
        super(Reasoner, self).__init__()
        self.bert = BertModel.from_pretrained( opt.PLM_name )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(opt.hidden_dim, opt.label_num) # 或者随便再麻烦点 TextCNN

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        # sequence_output = outputs[0] # 整个sequence; 就一bert输出
        sequence_output = outputs[1]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output) # 为判断是否 relevant # TODO: CE loss
        return logits  # (loss), scores, (hidden_states), (attentions)


# if __name__ == '__main__':
#     m = Reasoner()
#     print(m)