from transformers import BertTokenizer
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions, get_all_providers
import numpy as np

import sys
sys.path.append('.')
import os
from news_data.crawlNews.crawlNewPaper import crawl_News
import glob
from sklearn.metrics.pairwise import cosine_similarity
from underthesea import sent_tokenize
import re

class M_Sum():
    def create_model_for_provider(self,folder_model_path: str, provider: str) -> InferenceSession: 

        # Few properties that might have an impact on performances (provided by MS)
        options = SessionOptions()
        options.intra_op_num_threads = 1
        options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        model_path=''
        # Load the model as a graph and prepare the CPU backend 
        for file in os.listdir(folder_model_path):
            if file.endswith(".onnx"):
                model_path=os.path.join(folder_model_path,file)
            
        if model_path=='':
            return print("Could found model")
        session = InferenceSession(model_path, options, providers=[provider])
        session.disable_fallback()
            
        return session
    
    def __init__(self,lang='vi'):
        self.pretrained = "model/ViBert/vi_Bert_onnx" if lang=='vi' \
        else("model/ChBert/ch_Bert_onnx" if lang=='ch' \
        else ("model/RuBert/ru_Bert_onnx" if lang=='ru' \
        else 'model/EnBert/en_Bert_onnx'))
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained)
        self.cpu_model = self.create_model_for_provider(self.pretrained, "CPUExecutionProvider")
        
    def get_data_url(self, paper_url):
        title, description,paras = crawl_News(url=paper_url)
        return title, description, paras
            
    def vector_calculator_url(self, paper_url):
        title, description, paras = self.get_data_url(paper_url)
        # centroid vector
        # Inputs are provided through numpy array
        input_id_title = self.tokenizer(title, return_tensors="pt")
        inputs_title_onnx = {k: v.cpu().detach().numpy() for k, v in input_id_title.items()}
        # Run the model (None = get all the outputs)
        _, title_pooled = self.cpu_model.run(None, inputs_title_onnx)
        
        # Inputs are provided through numpy array
        input_id_description = self.tokenizer(description, return_tensors="pt")
        inputs_description_onnx = {k: v.cpu().detach().numpy() for k, v in input_id_description.items()}
        # Run the model (None = get all the outputs)
        _, description_pooled = self.cpu_model.run(None, inputs_description_onnx)
        
        t_d = np.stack((title_pooled, description_pooled))
        centroid_doc = np.mean(t_d, axis=0)
        
        #vector sentences
        n = len(paras)
        sents_vec_dict = {v: k for v, k in enumerate(paras)}    
        for index in range(n) : 
            input_id = self.tokenizer(paras[index], return_tensors="pt")
            inputs_onnx = {k: v.cpu().detach().numpy() for k, v in input_id.items()}
            # Run the model (None = get all the outputs)
            _, pooled = self.cpu_model.run(None, inputs_onnx)
            sents_vec_dict.update({index:pooled})
            
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
        doc = doc.replace('。','. ')
        doc = doc.replace('？','? ')
        doc = sent_tokenize(doc)
        print(doc)
        n = len(doc)
        sents_vec_dict = {v: k for v, k in enumerate(doc)}    
        for index in range(n) : 
            input_id = self.tokenizer(doc[index], return_tensors="pt")
            inputs_onnx = {k: v.cpu().detach().numpy() for k, v in input_id.items()}
            # Run the model (None = get all the outputs)
            _, pooled = self.cpu_model.run(None, inputs_onnx)
            sents_vec_dict.update({index:pooled})
            
        X = list(sents_vec_dict.values())
        X = np.stack(X)
        # print(X.shape)
        centroid_doc = np.mean(X,0)  
        return sents_vec_dict, centroid_doc
    def summary_doc(self, doc,k):
        # if auto_select_sent == False :
        #     k = 5
        # else :
        #     k = round(len(doc.split('.'))/ 2 + 1)
        doc = doc.replace('。','. ')
        doc = doc.replace('？','? ')
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
    def sum_main(self, news, k):
        """
        news : dictionary bao gom :
            title : title cua news
            description : None(gan bang title) or string, description cua news
            paras : list doan tin cua news
        k : % van ban tom tat
        """
        title = news['title']
        description = news['description']
        paras = news['paras']
        input_id_title = self.tokenizer(title, return_tensors="pt")
        inputs_title_onnx = {k: v.cpu().detach().numpy() for k, v in input_id_title.items()}
        # Run the model (None = get all the return sents_vec_dict, centroid_docoutputs)
        _, title_pooled = self.cpu_model.run(None, inputs_title_onnx)
        
        # Inputs are provided through numpy array
        input_id_description = self.tokenizer(description, return_tensors="pt")
        inputs_description_onnx = {k: v.cpu().detach().numpy() for k, v in input_id_description.items()}
        # Run the model (None = get all the outputs)
        _, description_pooled = self.cpu_model.run(None, inputs_description_onnx)
        
        t_d = np.stack((title_pooled, description_pooled))
        centroid_doc = np.mean(t_d, axis=0)
        
        #vector sentences
        n = len(paras)
        sents_vec_dict = {v: k for v, k in enumerate(paras)}    
        for index in range(n) : 
            input_id = self.tokenizer(paras[index], return_tensors="pt")
            inputs_onnx = {k: v.cpu().detach().numpy() for k, v in input_id.items()}
            # Run the model (None = get all the outputs)
            _, pooled = self.cpu_model.run(None, inputs_onnx)
            sents_vec_dict.update({index:pooled})
            
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


if __name__ == '__main__':    
    summ = M_Sum()
    news = {
        'title' : 'Ukraine khóa mũi tiến công của Nga ở Biển Đen trước trận đánh lớn',
        'description' : 'Ukraine tìm cách ngăn Nga kiểm soát Biển Đen trong bối cảnh Moscow được cho là chuẩn bị tiến hành cuộc tấn công quy mô lớn.',
        'paras' : 
            ['Khi được hỏi về nguy cơ tiềm ẩn đối với khu vực phía nam của Ukraine trước cuộc tấn công quy mô lớn sắp xảy ra của Nga, Bộ trưởng Quốc phòng Ukraine Oleksii Reznikov hôm 12/2 cho biết Ukraine tìm cách ngăn Nga kiểm soát Biển Đen - vùng biển chiến lược trong chiến dịch quân sự của Nga ở Ukraine.', 
             '"Tôi thực sự không thích đưa ra dự đoán hay đánh giá ý kiến, nhưng để kiểm soát Odessa và khu vực (phía nam) nói chung, Nga phải chiếm ưu thế trên Biển Đen. Tuy nhiên, chúng tôi đã tước đi cơ hội này của họ", Bộ trưởng Reznikov nói trong một cuộc họp báo.', 
             'Odessa là thành phố đông dân thứ 3 của Ukraine và là một trung tâm du lịch, thương mại lớn nằm trên bờ Tây Bắc Biển Đen. Odessa cũng là điểm trung chuyển lớn với 3 thương cảng, đồng thời là ngã ba đường sắt lớn nhất phía Nam Ukraine, do đó Odessa có ý nghĩa quan trọng chiến lược không chỉ về thương mại mà cả quy hoạch quân sự. Odessa cũng là nơi đặt Bộ Tư lệnh Hải quân của quân đội Ukraine.', 
             'Ông Reznikov đề cập đến việc Ukraine từng sử dụng Neptune, vũ khí chống hạm được sản xuất ở Ukraine, để nhắm vào tàu tuần dương Moskva của Nga hồi năm ngoái.', 
             '"Chúng tôi đã ngăn chặn sự thống trị của Nga ở Biển Đen, đặc biệt sau khi phóng thành công tên lửa Neptune khiến tàu tuần dương Moskva bị chìm tại khu vực này", ông Reznikov nói.', 
             'Vào tháng 4/2022, soái hạm Moskva thuộc Hạm đội Biển Đen của Nga bất ngờ bốc cháy ngoài khơi cách thành phố cảng Odessa của Ukraine khoảng 90km. Moscow tuyên bố một vụ hỏa hoạn do nổ kho đạn trên boong khiến con tàu hư hại nặng và bị đắm khi được lai dắt về cảng. Tuy nhiên, phía Ukraine khẳng định đã bắn cháy con tàu bằng hai tên lửa hành trình.', 
             'Theo Bộ trưởng Quốc phòng Ukraine, lực lượng Kiev cũng đang sử dụng tên lửa chống hạm Harpoon và có thể bảo vệ khu vực phía nam bằng vũ khí này.', 
             '"Các tổ hợp chống hạm Harpoon đang hoạt động, vì vậy tôi không thấy Nga có bất kỳ cơ hội nào để tiếp cận Odessa từ biển", ông Reznikov nhấn mạnh.', 
             'Bộ trưởng Reznikov nhận định Nga không thể kiểm soát Odessa bằng đường bộ. Ông cho biết lực lượng Nga đã bị đẩy lùi về bờ đông của sông Dnipro. "Đối với các vùng lãnh thổ tả ngạn sông ở miền nam Ukraine, lực lượng Nga ở đó có cơ hội bổ sung vũ khí, thiết bị và nhân lực, vì vậy tình hình ở đó tất nhiên căng thẳng hơn", ông nói thêm.', 
             'Trước đó, một số nguồn tin phương Tây tháng trước tiết\xa0lộ, hầu hết các tàu nổi và tàu ngầm của Hạm đội Biển Đen Nga đã rời căn cứ ở cảng Novorossiysk, miền Nam Nga. Các chuyên gia nhận định, Nga có thể sắp sử dụng hạm đội này cho một đợt tập kích quy mô lớn nhằm vào Ukraine.', 
             'Quân đội Ukraine dự đoán, Nga nhiều khả năng đang chuẩn bị cho một cuộc tấn công quy mô lớn vào tháng 2 hoặc tháng 3. Người phát ngôn Andriy Yusov của Tổng cục Tình báo Bộ Quốc phòng Ukraine cho rằng Tổng thống Nga Vladimir Putin đã ra lệnh cho Tổng tham mưu trưởng Lực lượng Vũ trang Nga, tướng Valeriy Gerasimov, kiểm soát hoàn toàn vùng Donbass, miền Đông Ukraine vào tháng 3.', 'Giới chức Ukraine khẳng định, họ đã sẵn sàng cho kịch bản tiến công của Nga, đồng thời hối thúc các đồng minh, đối tác phương Tây đẩy nhanh tốc độ viện trợ khí tài quân sự.']
    }
    r1 = summ.sum_main(news, 0.4)
    r2 = summ.summary_url('https://dantri.com.vn/the-gioi/ukraine-khoa-mui-tien-cong-cua-nga-o-bien-den-truoc-tran-danh-lon-20230213114746124.htm', 0.4)
    print(r1)
    print('----------------------------------------------------------------')
    print(r2)
    if r1 == r2:
        print('true')