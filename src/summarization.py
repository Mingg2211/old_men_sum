from transformers import *
import sys
sys.path.append('.')
import os
from news_data.crawlNews.crawlNewPaper import crawl_News
import glob
import json
import torch
from sklearn.metrics.pairwise import cosine_similarity
from underthesea import word_tokenize
from underthesea import sent_tokenize


class M_Sum():
    def __init__(self,lang='vi'):
        pretrained = "model/ViBert" if lang=='vi' else("                                                      model/ChBert/ch_Bert" if lang=='ch' else ("model/RuBert" if lang=='ru' else 'model/EnBert'))
        self.pretrained = pretrained
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained)
        self.model = BertModel.from_pretrained(self.pretrained)

    def get_data_url(self, paper_url):
        title, description,paras = crawl_News(url=paper_url)
        return title, description, paras
            
    def vector_calculator_url(self, paper_url):
        title, description, paras = self.get_data_url(paper_url)
        # centroid vector
        input_id_title = self.tokenizer.encode(title,add_special_tokens = True)
        att_mask_title = [int(token_id > 0) for token_id in input_id_title]
        
        
        
        print(type([input_id_title]))
        input_ids_title = torch.tensor([input_id_title])
        att_masks_title = torch.tensor([att_mask_title])
        input_id_description = self.tokenizer.encode(description,add_special_tokens = True)
        att_mask_description = [int(token_id > 0) for token_id in input_id_description]
        input_ids_description = torch.tensor([input_id_description])
        att_masks_description = torch.tensor([att_mask_description])
        
        with torch.no_grad():
            features_title = self.model(input_ids_title,att_masks_title)
            features_description = self.model(input_ids_description,att_masks_description)
        t_d = torch.stack((features_title.pooler_output, features_description.pooler_output))
        centroid_doc = torch.mean(t_d, axis=0)
        # print(centroid_doc.shape)
        
        #vector sentences
        n = len(paras)
        sents_vec_dict = {v: k for v, k in enumerate(paras)}    
        for index in range(n) : 
            input_id = self.tokenizer.encode(paras[index],add_special_tokens = True)
            att_mask = [int(token_id > 0) for token_id in input_id]
            input_ids = torch.tensor([input_id])
            att_masks = torch.tensor([att_mask])
            with torch.no_grad():
                features = self.model(input_ids,att_masks)
            sents_vec_dict.update({index:features.pooler_output})
            
        return sents_vec_dict, centroid_doc

    def summary_url(self,paper_url, k):
        paras = self.get_data_url(paper_url)[2]
        sents_vec_dict, centroid_doc = self.vector_calculator_url(paper_url)
        cosine_sim = {}
        for key in sents_vec_dict.keys():
            cosine_2vec = cosine_similarity(centroid_doc, sents_vec_dict[key])
            cosine_sim.update({key:cosine_2vec})
        final_sim = sorted(cosine_sim.items(), key=lambda x:x[1], reverse=True)
        chossen = round(k*len(final_sim))
        list_index = dict(final_sim[:chossen]).keys()
        # print(list_index)
        result = []
        for index in sorted(list_index):
            result.append(paras[index])
        return '\n\n'.join(result)
    def vector_calculator_doc(self, doc):
        doc_tok = [word_tokenize(doc, format='text') for doc in sent_tokenize(doc)]
        n = len(doc_tok)
        sents_vec_dict = {v: k for v, k in enumerate(doc_tok)}    
        for index in range(n) : 
            input_id = self.tokenizer.encode(doc_tok[index],add_special_tokens = True)
            att_mask = [int(token_id > 0) for token_id in input_id]
            input_ids = torch.tensor([input_id])
            att_masks = torch.tensor([att_mask])
            with torch.no_grad():
                features = self.model(input_ids,att_masks)
            sents_vec_dict.update({index:features.pooler_output})
        X = list(sents_vec_dict.values())
        X = torch.stack(X)
        # print(X.shape)
        centroid_doc = torch.mean(X,0)  
        return sents_vec_dict, centroid_doc
    def summary_doc(self, doc,k):
        # if auto_select_sent == False :
        #     k = 5
        # else :
        #     k = round(len(doc.split('.'))/ 2 + 1)
        doc_sents = sent_tokenize(doc)
        sents_vec_dict, centroid_doc = self.vector_calculator_doc(doc)
        cosine_sim = {}
        for key in sents_vec_dict.keys():
            cosine_2vec = cosine_similarity(centroid_doc, sents_vec_dict[key])
            # print(centroid_doc.shape, sents_vec_dict[key].shape)
            cosine_sim.update({key:cosine_2vec})
        final_sim = sorted(cosine_sim.items(), key=lambda x:x[1], reverse=True)
        chossen = round(k*len(final_sim))
        
        list_index = dict(final_sim[:chossen]).keys()
        result = []
        
        for index in sorted(list_index):
            result.append(doc_sents[index])
        mingg = '\n\n'.join(result)
        return mingg

    
if __name__ == '__main__':    
    sum = M_Sum('ch')  
    print(sum.summary_url('https://cn.chinadaily.com.cn/a/202302/21/WS63f42ccca3102ada8b22fe7b.html'))
    # sum = M_Sum('ru')  
    # print(sum.summary_url('https://www.mosobl.kp.ru/daily/27398.5/4594049/'))