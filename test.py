import math
import pandas as pd
import numpy as np
import torch
# import shap
from torch import nn
from tqdm import tqdm
# from sklearn.metrics import classification_report, accuracy_score
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import TensorDataset, DataLoader
from pytorch_pretrained_bert import BertTokenizer, BertModel
# from pytorch_pretrained_bert.optimization import BertAdam
# from matplotlib import pyplot as plt
import torch.nn.functional as F
# from torchviz import make_dot


class ClassifyModel(nn.Module):
    def __init__(self, pretrained_model_name_or_path, num_labels, batch_size, is_lock=False):
        super(ClassifyModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)
        config = self.bert.config
        if is_lock:
            for name, param in self.bert.named_parameters():
                if name.startswith('pooler'):
                    continue
                else:
                    param.requires_grad_(False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_L1 = nn.Linear(7680, num_labels)
        self.w_gcb_p = nn.Parameter(nn.init.xavier_uniform_(torch.ones(768, 768)))
        self.w_gca_p = nn.Parameter(nn.init.xavier_uniform_(torch.ones(768, 768)))
        self.w_fcb_p = nn.Parameter(nn.init.xavier_uniform_(torch.ones(1536, 768)))
        self.w_fca_p = nn.Parameter(nn.init.xavier_uniform_(torch.ones(1536, 768)))
        self.w_gsb_p = nn.Parameter(nn.init.xavier_uniform_(torch.ones(768, 768)))
        self.w_gsa_p = nn.Parameter(nn.init.xavier_uniform_(torch.ones(768, 768)))
        self.w_fsb_p = nn.Parameter(nn.init.xavier_uniform_(torch.ones(1536, 768)))
        self.w_fsa_p = nn.Parameter(nn.init.xavier_uniform_(torch.ones(1536, 768)))
        self.w_tempp = nn.Parameter(nn.init.xavier_uniform_(torch.ones(1536, 768)))
        self.batch_norm_temp0 = nn.BatchNorm1d(768)
        self.batch_norm_temp2 = nn.BatchNorm1d(768)
        self.batch_norm_tmp0 = nn.BatchNorm1d(768)
        self.batch_norm_tmp2 = nn.BatchNorm1d(768)

        self.w_gcb_c = nn.Parameter(nn.init.xavier_uniform_(torch.ones(768, 768)))
        self.w_gca_c = nn.Parameter(nn.init.xavier_uniform_(torch.ones(768, 768)))
        self.w_fcb_c = nn.Parameter(nn.init.xavier_uniform_(torch.ones(1536, 768)))
        self.w_fca_c = nn.Parameter(nn.init.xavier_uniform_(torch.ones(1536, 768)))
        self.w_gsb_c = nn.Parameter(nn.init.xavier_uniform_(torch.ones(768, 768)))
        self.w_gsa_c = nn.Parameter(nn.init.xavier_uniform_(torch.ones(768, 768)))
        self.w_fsb_c = nn.Parameter(nn.init.xavier_uniform_(torch.ones(1536, 768)))
        self.w_fsa_c = nn.Parameter(nn.init.xavier_uniform_(torch.ones(1536, 768)))
        self.w_tempc = nn.Parameter(nn.init.xavier_uniform_(torch.ones(1536, 768)))
        self.batch_norm_temp0c = nn.BatchNorm1d(768)
        self.batch_norm_temp2c = nn.BatchNorm1d(768)
        self.batch_norm_tmp0c = nn.BatchNorm1d(768)
        self.batch_norm_tmp2c = nn.BatchNorm1d(768)

        self.w_r = nn.Parameter(nn.init.xavier_uniform_(torch.ones(3072, 1536)))
        self.w_z = nn.Parameter(nn.init.xavier_uniform_(torch.ones(3072, 1536)))
        self.w_h = nn.Parameter(nn.init.xavier_uniform_(torch.ones(3072, 1536)))
        self.batch_norm_r = nn.BatchNorm1d(1536)
        self.batch_norm_z = nn.BatchNorm1d(1536)
        self.w_com1 = nn.Parameter(nn.init.xavier_uniform_(torch.ones(3072, 1536)))
        self.w_com2 = nn.Parameter(nn.init.xavier_uniform_(torch.ones(3072, 1536)))
        self.w_com3 = nn.Parameter(nn.init.xavier_uniform_(torch.ones(3072, 1536)))
        self.w_com4 = nn.Parameter(nn.init.xavier_uniform_(torch.ones(3072, 1536)))
        self.w_com5 = nn.Parameter(nn.init.xavier_uniform_(torch.ones(3072, 1536)))

    def HGD_FN_Post(self, message_1, message_2):
        phi_ci = F.leaky_relu(torch.cat(
            (torch.matmul(message_1, self.w_gcb_p), torch.matmul(message_2, self.w_gca_p)), dim=1), negative_slope=0.01)
        # print("Grad_fn of phi_ci: ", phi_ci.grad_fn)

        temp0 = torch.matmul(phi_ci, self.w_fcb_p)
        temp0 = self.batch_norm_temp0(temp0)  # Apply Batch Normalization
        # print("Grad_fn of temp0: ", temp0.grad_fn)

        temp1 = torch.mul(message_1, temp0)
        # print("Grad_fn of temp1: ", temp1.grad_fn)

        temp2 = torch.matmul((1 - phi_ci), self.w_fca_p)
        temp2 = self.batch_norm_temp2(temp2)  # Apply Batch Normalization
        # print("Grad_fn of temp2: ", temp2.grad_fn)

        temp3 = torch.mul(message_2, temp2)
        # print("Grad_fn of temp3: ", temp3.grad_fn)

        F_Ci = torch.tanh(temp1 + temp3)
        # print("Grad_fn of F_Ci: ", F_Ci.grad_fn)

        phi_si = F.leaky_relu(torch.cat(
            (torch.matmul(message_1, self.w_gsb_p), torch.matmul(message_2, self.w_gsa_p)), dim=1), negative_slope=0.01)
        # print("Grad_fn of phi_si: ", phi_si.grad_fn)

        tmp0 = torch.matmul(phi_si, self.w_fsb_p)
        tmp0 = self.batch_norm_tmp0(tmp0)  # Apply Batch Normalization
        # print("Grad_fn of tmp0: ", tmp0.grad_fn)

        tmp1 = torch.mul(message_1, tmp0)
        # print("Grad_fn of tmp1: ", tmp1.grad_fn)

        tmp2 = torch.matmul(phi_si, self.w_fsa_p)
        tmp2 = self.batch_norm_tmp2(tmp2)  # Apply Batch Normalization
        # print("Grad_fn of tmp2: ", tmp2.grad_fn)

        tmp3 = torch.mul(message_2, tmp2)
        # print("Grad_fn of tmp3: ", tmp3.grad_fn)

        F_Si = torch.tanh(tmp1 + tmp3)
        # print("Grad_fn of F_Si: ", F_Si.grad_fn)

        temppp = torch.matmul(phi_si, self.w_tempp)
        # print("Grad_fn of temppp: ", temppp.grad_fn)

        m_ip = torch.cat((F_Si, torch.mul(F_Ci, (1 - temppp))), dim=1)
        # print("Grad_fn of m_i: ", m_ip.grad_fn)

        return m_ip


    def HGD_FN_comment(self, com_n, com_n_n):
        phi_ci = F.leaky_relu(torch.cat(
            (torch.matmul(com_n, self.w_gcb_c), torch.matmul(com_n_n, self.w_gca_c)), dim=1), negative_slope=0.01)
        # print("Grad_fn of phi_ci: ", phi_ci.grad_fn)

        temp0 = torch.matmul(phi_ci, self.w_fcb_c)
        temp0 = self.batch_norm_temp0c(temp0)  # Apply Batch Normalization
        # print("Grad_fn of temp0: ", temp0.grad_fn)

        temp1 = torch.mul(com_n, temp0)
        # print("Grad_fn of temp1: ", temp1.grad_fn)

        temp2 = torch.matmul((1 - phi_ci), self.w_fca_c)
        temp2 = self.batch_norm_temp2c(temp2)  # Apply Batch Normalization
        # print("Grad_fn of temp2: ", temp2.grad_fn)

        temp3 = torch.mul(com_n_n, temp2)
        # print("Grad_fn of temp3: ", temp3.grad_fn)

        F_Ci = torch.tanh(temp1 + temp3)
        # print("Grad_fn of F_Ci: ", F_Ci.grad_fn)

        phi_si = F.leaky_relu(torch.cat(
            (torch.matmul(com_n, self.w_gsb_c), torch.matmul(com_n_n, self.w_gsa_c)), dim=1), negative_slope=0.01)
        # print("Grad_fn of phi_si: ", phi_si.grad_fn)

        tmp0 = torch.matmul(phi_si, self.w_fsb_c)
        tmp0 = self.batch_norm_tmp0c(tmp0)  # Apply Batch Normalization
        # print("Grad_fn of tmp0: ", tmp0.grad_fn)

        tmp1 = torch.mul(com_n, tmp0)
        # print("Grad_fn of tmp1: ", tmp1.grad_fn)

        tmp2 = torch.matmul(phi_si, self.w_fsa_c)
        tmp2 = self.batch_norm_tmp2c(tmp2)  # Apply Batch Normalization
        # print("Grad_fn of tmp2: ", tmp2.grad_fn)

        tmp3 = torch.mul(com_n_n, tmp2)
        # print("Grad_fn of tmp3: ", tmp3.grad_fn)

        F_Si = torch.tanh(tmp1 + tmp3)
        # print("Grad_fn of F_Si: ", F_Si.grad_fn)

        tempcp = torch.matmul(phi_si, self.w_tempc)
        # print("Grad_fn of temppp: ", temppp.grad_fn)

        m_ic = torch.cat((F_Si, torch.mul(F_Ci, (1 - tempcp))), dim=1)
        # print("Grad_fn of m_i: ", m_ic.grad_fn)

        return m_ic


    def SGRU(self, Xt, Ht_1):
        concatenated = torch.cat((Xt, Ht_1), dim=1)
        Rt = torch.sigmoid(self.batch_norm_r(torch.matmul(concatenated, self.w_r)))
        Zt = torch.sigmoid(self.batch_norm_z(torch.matmul(concatenated, self.w_z)))  # batch_size, 1536
        Ht_hidden = torch.tanh(torch.matmul(torch.cat((Xt, Rt * Ht_1), dim=1), self.w_h))
        Ht = Zt * Ht_1 + (1 - Zt) * Ht_hidden
        return Ht

    def forward(self, batch_Data):
        token_type_ids = None
        attention_mask = None
        batch_Data = batch_Data.transpose(0, 1).to(device)
        input_post = batch_Data[0].to(torch.int64)
        input_com1 = batch_Data[1].to(torch.int64)
        input_com11 = batch_Data[2].to(torch.int64)
        input_com12 = batch_Data[3].to(torch.int64)
        input_com13 = batch_Data[4].to(torch.int64)
        input_com2 = batch_Data[5].to(torch.int64)
        input_com21 = batch_Data[6].to(torch.int64)
        input_com22 = batch_Data[7].to(torch.int64)
        input_com23 = batch_Data[8].to(torch.int64)
        input_com3 = batch_Data[9].to(torch.int64)
        input_com31 = batch_Data[10].to(torch.int64)
        input_com32 = batch_Data[11].to(torch.int64)
        input_com33 = batch_Data[12].to(torch.int64)
        input_com4 = batch_Data[13].to(torch.int64)
        input_com41 = batch_Data[14].to(torch.int64)
        input_com42 = batch_Data[15].to(torch.int64)
        input_com43 = batch_Data[16].to(torch.int64)
        input_com5 = batch_Data[17].to(torch.int64)
        input_com51 = batch_Data[18].to(torch.int64)
        input_com52 = batch_Data[19].to(torch.int64)
        input_com53 = batch_Data[20].to(torch.int64)

        _, pooled_post = self.bert(input_post, token_type_ids, attention_mask, output_all_encoded_layers=False)
        _, pooled_com1 = self.bert(input_com1, token_type_ids, attention_mask, output_all_encoded_layers=False)
        _, pooled_com11 = self.bert(input_com11, token_type_ids, attention_mask, output_all_encoded_layers=False)
        _, pooled_com12 = self.bert(input_com12, token_type_ids, attention_mask, output_all_encoded_layers=False)
        _, pooled_com13 = self.bert(input_com13, token_type_ids, attention_mask, output_all_encoded_layers=False)

        _, pooled_com2 = self.bert(input_com2, token_type_ids, attention_mask, output_all_encoded_layers=False)
        _, pooled_com21 = self.bert(input_com21, token_type_ids, attention_mask, output_all_encoded_layers=False)
        _, pooled_com22 = self.bert(input_com22, token_type_ids, attention_mask, output_all_encoded_layers=False)
        _, pooled_com23 = self.bert(input_com23, token_type_ids, attention_mask, output_all_encoded_layers=False)

        _, pooled_com3 = self.bert(input_com3, token_type_ids, attention_mask, output_all_encoded_layers=False)
        _, pooled_com31 = self.bert(input_com31, token_type_ids, attention_mask, output_all_encoded_layers=False)
        _, pooled_com32 = self.bert(input_com32, token_type_ids, attention_mask, output_all_encoded_layers=False)
        _, pooled_com33 = self.bert(input_com33, token_type_ids, attention_mask, output_all_encoded_layers=False)

        _, pooled_com4 = self.bert(input_com4, token_type_ids, attention_mask, output_all_encoded_layers=False)
        _, pooled_com41 = self.bert(input_com41, token_type_ids, attention_mask, output_all_encoded_layers=False)
        _, pooled_com42 = self.bert(input_com42, token_type_ids, attention_mask, output_all_encoded_layers=False)
        _, pooled_com43 = self.bert(input_com43, token_type_ids, attention_mask, output_all_encoded_layers=False)

        _, pooled_com5 = self.bert(input_com5, token_type_ids, attention_mask, output_all_encoded_layers=False)
        _, pooled_com51 = self.bert(input_com51, token_type_ids, attention_mask, output_all_encoded_layers=False)
        _, pooled_com52 = self.bert(input_com52, token_type_ids, attention_mask, output_all_encoded_layers=False)
        _, pooled_com53 = self.bert(input_com53, token_type_ids, attention_mask, output_all_encoded_layers=False)

        contrast_p_1 = self.HGD_FN_Post(pooled_post, pooled_com1)
        contrast_p_2 = self.HGD_FN_Post(pooled_post, pooled_com2)
        contrast_p_3 = self.HGD_FN_Post(pooled_post, pooled_com3)
        contrast_p_4 = self.HGD_FN_Post(pooled_post, pooled_com4)
        contrast_p_5 = self.HGD_FN_Post(pooled_post, pooled_com5)

        contrast_c1_1 = self.HGD_FN_comment(pooled_com1, pooled_com11)
        contrast_c1_2 = self.HGD_FN_comment(pooled_post, pooled_com12)
        contrast_c1_3 = self.HGD_FN_comment(pooled_post, pooled_com13)
        Ht = model.SGRU(contrast_c1_2, contrast_c1_1)
        HT_1 = model.SGRU(contrast_c1_3, Ht)
        contrast_c2_1 = self.HGD_FN_comment(pooled_com2, pooled_com21)
        contrast_c2_2 = self.HGD_FN_comment(pooled_com2, pooled_com22)
        contrast_c2_3 = self.HGD_FN_comment(pooled_com2, pooled_com23)
        Ht = model.SGRU(contrast_c2_2, contrast_c2_1)
        HT_2 = model.SGRU(contrast_c2_3, Ht)
        contrast_c3_1 = self.HGD_FN_comment(pooled_com3, pooled_com31)
        contrast_c3_2 = self.HGD_FN_comment(pooled_com3, pooled_com32)
        contrast_c3_3 = self.HGD_FN_comment(pooled_com3, pooled_com33)
        Ht = model.SGRU(contrast_c3_2, contrast_c3_1)
        HT_3 = model.SGRU(contrast_c3_3, Ht)
        contrast_c4_1 = self.HGD_FN_comment(pooled_com4, pooled_com41)
        contrast_c4_2 = self.HGD_FN_comment(pooled_com4, pooled_com42)
        contrast_c4_3 = self.HGD_FN_comment(pooled_com4, pooled_com43)
        Ht = model.SGRU(contrast_c4_2, contrast_c4_1)
        HT_4 = model.SGRU(contrast_c4_3, Ht)
        contrast_c5_1 = self.HGD_FN_comment(pooled_com5, pooled_com51)
        contrast_c5_2 = self.HGD_FN_comment(pooled_com5, pooled_com52)
        contrast_c5_3 = self.HGD_FN_comment(pooled_com5, pooled_com53)
        Ht = model.SGRU(contrast_c5_2, contrast_c5_1)
        HT_5 = model.SGRU(contrast_c5_3, Ht)

        combine_1 = torch.sigmoid(torch.matmul(torch.cat((contrast_p_1, HT_1), dim=1), self.w_com1))
        combine_2 = torch.sigmoid(torch.matmul(torch.cat((contrast_p_2, HT_2), dim=1), self.w_com2))
        combine_3 = torch.sigmoid(torch.matmul(torch.cat((contrast_p_3, HT_3), dim=1), self.w_com3))
        combine_4 = torch.sigmoid(torch.matmul(torch.cat((contrast_p_4, HT_4), dim=1), self.w_com4))
        combine_5 = torch.sigmoid(torch.matmul(torch.cat((contrast_p_5, HT_5), dim=1), self.w_com5))

        result1 = torch.cat((combine_1, combine_2, combine_3, combine_4, combine_5), dim=1)
        result2 = self.dropout(result1)
        result3 = self.output_L1(result2)

        # make_dot(result3, params=dict(self.named_parameters())).render("model_structure", format="png")

        return result3


def print_grad(grad):
    if torch.all(grad == 0):
        print(grad)


class DataProcessForMultipleSentences(object):
    def __init__(self, bert_tokenizer, max_workers=10):
        self.bert_tokenizer = bert_tokenizer
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        tokenizer_params = {
            'padding': 'max_length',
            'max_length': 512,
            'truncation': True,
            'return_tensors': 'pt'
        }

    def get_input(self, dataset, max_seq_len=512):
        message_post = dataset.iloc[:, 0].tolist()
        com1 = dataset.iloc[:, 2].tolist()
        com11 = dataset.iloc[:, 4].tolist()
        com12 = dataset.iloc[:, 6].tolist()
        com13 = dataset.iloc[:, 8].tolist()
        com2 = dataset.iloc[:, 10].tolist()
        com21 = dataset.iloc[:, 12].tolist()
        com22 = dataset.iloc[:, 14].tolist()
        com23 = dataset.iloc[:, 16].tolist()
        com3 = dataset.iloc[:, 18].tolist()
        com31 = dataset.iloc[:, 20].tolist()
        com32 = dataset.iloc[:, 22].tolist()
        com33 = dataset.iloc[:, 24].tolist()
        com4 = dataset.iloc[:, 26].tolist()
        com41 = dataset.iloc[:, 28].tolist()
        com42 = dataset.iloc[:, 30].tolist()
        com43 = dataset.iloc[:, 32].tolist()
        com5 = dataset.iloc[:, 34].tolist()
        com51 = dataset.iloc[:, 36].tolist()
        com52 = dataset.iloc[:, 38].tolist()
        com53 = dataset.iloc[:, 40].tolist()
        vague_labels = dataset.iloc[:, 1].tolist()

        message_post = list(self.pool.map(self.bert_tokenizer.tokenize, message_post))
        com1 = list(self.pool.map(self.bert_tokenizer.tokenize, com1))
        com11 = list(self.pool.map(self.bert_tokenizer.tokenize, com11))
        com12 = list(self.pool.map(self.bert_tokenizer.tokenize, com12))
        com13 = list(self.pool.map(self.bert_tokenizer.tokenize, com13))

        com2 = list(self.pool.map(self.bert_tokenizer.tokenize, com2))
        com21 = list(self.pool.map(self.bert_tokenizer.tokenize, com21))
        com22 = list(self.pool.map(self.bert_tokenizer.tokenize, com22))
        com23 = list(self.pool.map(self.bert_tokenizer.tokenize, com23))

        com3 = list(self.pool.map(self.bert_tokenizer.tokenize, com3))
        com31 = list(self.pool.map(self.bert_tokenizer.tokenize, com31))
        com32 = list(self.pool.map(self.bert_tokenizer.tokenize, com32))
        com33 = list(self.pool.map(self.bert_tokenizer.tokenize, com33))

        com4 = list(self.pool.map(self.bert_tokenizer.tokenize, com4))
        com41 = list(self.pool.map(self.bert_tokenizer.tokenize, com41))
        com42 = list(self.pool.map(self.bert_tokenizer.tokenize, com42))
        com43 = list(self.pool.map(self.bert_tokenizer.tokenize, com43))

        com5 = list(self.pool.map(self.bert_tokenizer.tokenize, com5))
        com51 = list(self.pool.map(self.bert_tokenizer.tokenize, com51))
        com52 = list(self.pool.map(self.bert_tokenizer.tokenize, com52))
        com53 = list(self.pool.map(self.bert_tokenizer.tokenize, com53))

        message_post = list(self.pool.map(self.trunate_and_pad, message_post, [max_seq_len] * len(message_post)))
        com1 = list(self.pool.map(self.trunate_and_pad, com1, [max_seq_len] * len(com1)))
        com11 = list(self.pool.map(self.trunate_and_pad, com11, [max_seq_len] * len(com11)))
        com12 = list(self.pool.map(self.trunate_and_pad, com12, [max_seq_len] * len(com12)))
        com13 = list(self.pool.map(self.trunate_and_pad, com13, [max_seq_len] * len(com13)))

        com2 = list(self.pool.map(self.trunate_and_pad, com2, [max_seq_len] * len(com2)))
        com21 = list(self.pool.map(self.trunate_and_pad, com21, [max_seq_len] * len(com21)))
        com22 = list(self.pool.map(self.trunate_and_pad, com22, [max_seq_len] * len(com22)))
        com23 = list(self.pool.map(self.trunate_and_pad, com23, [max_seq_len] * len(com23)))

        com3 = list(self.pool.map(self.trunate_and_pad, com3, [max_seq_len] * len(com3)))
        com31 = list(self.pool.map(self.trunate_and_pad, com31, [max_seq_len] * len(com31)))
        com32 = list(self.pool.map(self.trunate_and_pad, com32, [max_seq_len] * len(com32)))
        com33 = list(self.pool.map(self.trunate_and_pad, com33, [max_seq_len] * len(com33)))

        com4 = list(self.pool.map(self.trunate_and_pad, com4, [max_seq_len] * len(com4)))
        com41 = list(self.pool.map(self.trunate_and_pad, com41, [max_seq_len] * len(com41)))
        com42 = list(self.pool.map(self.trunate_and_pad, com42, [max_seq_len] * len(com42)))
        com43 = list(self.pool.map(self.trunate_and_pad, com43, [max_seq_len] * len(com43)))
        com5 = list(self.pool.map(self.trunate_and_pad, com5, [max_seq_len] * len(com5)))
        com51 = list(self.pool.map(self.trunate_and_pad, com51, [max_seq_len] * len(com51)))
        com52 = list(self.pool.map(self.trunate_and_pad, com52, [max_seq_len] * len(com52)))
        com53 = list(self.pool.map(self.trunate_and_pad, com53, [max_seq_len] * len(com53)))

        seqs_post = [i[0] for i in message_post]
        seqs_com1 = [i[0] for i in com1]
        seqs_com11 = [i[0] for i in com11]
        seqs_com12 = [i[0] for i in com12]
        seqs_com13 = [i[0] for i in com13]

        seqs_com2 = [i[0] for i in com2]
        seqs_com21 = [i[0] for i in com21]
        seqs_com22 = [i[0] for i in com22]
        seqs_com23 = [i[0] for i in com23]

        seqs_com3 = [i[0] for i in com3]
        seqs_com31 = [i[0] for i in com31]
        seqs_com32 = [i[0] for i in com32]
        seqs_com33 = [i[0] for i in com33]

        seqs_com4 = [i[0] for i in com4]
        seqs_com41 = [i[0] for i in com41]
        seqs_com42 = [i[0] for i in com42]
        seqs_com43 = [i[0] for i in com43]
        seqs_com5 = [i[0] for i in com5]
        seqs_com51 = [i[0] for i in com51]
        seqs_com52 = [i[0] for i in com52]
        seqs_com53 = [i[0] for i in com53]

        t_seqs_post = torch.tensor(seqs_post, dtype=torch.long)
        t_seqs_com1 = torch.tensor(seqs_com1, dtype=torch.long)
        t_seqs_com11 = torch.tensor(seqs_com11, dtype=torch.long)
        t_seqs_com12 = torch.tensor(seqs_com12, dtype=torch.long)
        t_seqs_com13 = torch.tensor(seqs_com13, dtype=torch.long)

        t_seqs_com2 = torch.tensor(seqs_com2, dtype=torch.long)
        t_seqs_com21 = torch.tensor(seqs_com21, dtype=torch.long)
        t_seqs_com22 = torch.tensor(seqs_com22, dtype=torch.long)
        t_seqs_com23 = torch.tensor(seqs_com23, dtype=torch.long)

        t_seqs_com3 = torch.tensor(seqs_com3, dtype=torch.long)
        t_seqs_com31 = torch.tensor(seqs_com31, dtype=torch.long)
        t_seqs_com32 = torch.tensor(seqs_com32, dtype=torch.long)
        t_seqs_com33 = torch.tensor(seqs_com33, dtype=torch.long)

        t_seqs_com4 = torch.tensor(seqs_com4, dtype=torch.long)
        t_seqs_com41 = torch.tensor(seqs_com41, dtype=torch.long)
        t_seqs_com42 = torch.tensor(seqs_com42, dtype=torch.long)
        t_seqs_com43 = torch.tensor(seqs_com43, dtype=torch.long)
        t_seqs_com5 = torch.tensor(seqs_com5, dtype=torch.long)
        t_seqs_com51 = torch.tensor(seqs_com51, dtype=torch.long)
        t_seqs_com52 = torch.tensor(seqs_com52, dtype=torch.long)
        t_seqs_com53 = torch.tensor(seqs_com53, dtype=torch.long)

        t_labels = torch.tensor(vague_labels, dtype=torch.long)

        return TensorDataset(t_seqs_post, t_seqs_com1, t_seqs_com11, t_seqs_com12, t_seqs_com13, t_seqs_com2, t_seqs_com21, t_seqs_com22, t_seqs_com23, t_seqs_com3, t_seqs_com31, t_seqs_com32, t_seqs_com33, t_seqs_com4, t_seqs_com41, t_seqs_com42, t_seqs_com43, t_seqs_com5, t_seqs_com51, t_seqs_com52, t_seqs_com53, t_labels)

    def trunate_and_pad(self, seq, max_seq_len):
        if len(seq) > (max_seq_len - 2):
            seq = seq[0: (max_seq_len - 2)]
        seq = ['[CLS]'] + seq + ['[SEP]']
        seq = self.bert_tokenizer.convert_tokens_to_ids(seq)
        padding = [0] * (max_seq_len - len(seq))
        seq_mask = [1] * len(seq) + padding
        seq_segment = [0] * len(seq) + padding
        seq += padding
        assert len(seq) == max_seq_len
        assert len(seq_mask) == max_seq_len
        assert len(seq_segment) == max_seq_len
        return seq, seq_mask, seq_segment


def load_data(filepath, pretrained_model_name_or_path, max_seq_len, batch_size):
    io = pd.io.excel.ExcelFile(filepath)
    raw_train_data = pd.read_excel(io, sheet_name='train')
    raw_test_data = pd.read_excel(io, sheet_name='Test_ALL')
    io.close()
    bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path, do_lower_case=True)
    processor = DataProcessForMultipleSentences(bert_tokenizer=bert_tokenizer)
    train_data = processor.get_input(raw_train_data, max_seq_len)
    test_data = processor.get_input(raw_test_data, max_seq_len)

    train_iter = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    total_train_batch = math.ceil(len(raw_train_data) / batch_size)
    total_test_batch = math.ceil(len(raw_test_data) / batch_size)
    return train_iter, test_iter, total_train_batch, total_test_batch


from sklearn.metrics import classification_report, accuracy_score


def evaluate_accuracy(train_iter, data_iter, net, device, batch_count):
    prediction_labels, true_labels = [], []

    with torch.no_grad():
        for batch_data in tqdm(data_iter, desc='eval', total=batch_count):
            labels = batch_data[-1]
            labels = torch.tensor(labels).to(device)
            data_array_test = batch_data[:21]
            batch_list_numpy_test = [[tensor.numpy() for tensor in sublist] for sublist in data_array_test]
            batch_array_test = np.array(batch_list_numpy_test, dtype=np.float32)
            batch_array_test = np.transpose(batch_array_test, (1, 0, 2))
            batch_tensor_test = torch.from_numpy(batch_array_test).to(device)

            output = net(batch_tensor_test)
            predictions = output.softmax(dim=1).argmax(dim=1)

            prediction_labels.append(predictions.detach().cpu().numpy())
            true_labels.append(labels.detach().cpu().numpy())

    report = classification_report(np.concatenate(true_labels), np.concatenate(prediction_labels), digits=4)
    accuracy = accuracy_score(np.concatenate(true_labels), np.concatenate(prediction_labels))

    return report, accuracy


if __name__ == '__main__':
    batch_size, max_seq_len = 32, 512
    train_iter, test_iter, train_batch_count, test_batch_count = load_data('Dataset.xlsx',
                                                                           'bert-base-chinese', max_seq_len, batch_size)
    model_instance = torch.load('Model_CAPC.bin')
    state_dict = model_instance.state_dict()
    model = ClassifyModel('bert-base-chinese', num_labels=2, batch_size=batch_size, is_lock=True)
    model.load_state_dict(state_dict, strict=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    result, test_acc = evaluate_accuracy(train_iter, test_iter, model, device, test_batch_count)

    print(result)
    print(test_acc)
