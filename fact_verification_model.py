"""
Class allowing for the fine-tuning of Bert and Roberta

Freezes all of Bert's predefined weights, and just tunes final matrix of weights

Inspiration taken from: https://medium.com/swlh/painless-fine-tuning-of-bert-in-pytorch-b91c14912caa
"""
import torch
import torch.nn as nn
from transformers import BertModel, RobertaModel

HIDDEN_STATE_DIM = 768

"""
    The idea here being, lets just assume the hidden vector of the last layer of the CLS token from a
    BERT based model is somewhat reliable. in terms of being a good encoder of information. We 
    then take that encoding and train a single linear projection weights to do the final classification for us.
    So instead of tuning all the weights we just tune the single linear projection weights. For overfitting
    reasons I added a small dropout layer as well.

    Can handle both Roberta and Bert models
"""
class FactVerificationClf(nn.Module):
    def __init__(self, model_name, model_type="bert", freeze_model=True, dropout_pct=0.2, num_classes=2):
        super(FactVerificationClf, self).__init__()

        # loading correct pre-trained model
        if model_type == "bert":
            self.encoding_layer = BertModel.from_pretrained(model_name)
        else:
            self.encoding_layer = RobertaModel.from_pretrained(model_name)

        # "freezes" all parameters associated with BERT
        if freeze_model:
            for p in self.encoding_layer.parameters():
                p.requires_grad = False

        self.dropout = nn.Dropout(dropout_pct)

        self.output_matrix = nn.Linear(HIDDEN_STATE_DIM, num_classes)

    def forward(self, input_ids, input_mask, segment_ids):

        all_hidden_states, _ = self.encoding_layer(input_ids, input_mask, segment_ids)

        cls_hidden_state = all_hidden_states[:, 0]

        cls_hidden_state = self.dropout(cls_hidden_state)

        scores = self.output_matrix(cls_hidden_state)

        # return scores
        return torch.log_softmax(scores, dim=1)

