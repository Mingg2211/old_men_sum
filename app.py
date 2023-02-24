import streamlit as st
from src.sum_onnx import M_Sum
import os

@st.cache_resource
def load_model():
    vi_sum = M_Sum('vi')
    ru_sum = M_Sum('ru')
    en_sum = M_Sum('en')
    ch_sum = M_Sum('ch')
    return vi_sum,ru_sum,en_sum,ch_sum
vi_sum,ru_sum,en_sum,ch_sum = load_model()
#streamlit
st.title("AIA M_Summarizer")
st.subheader("Paste any article in text area below and click on the 'Summarize Text' button to get the summarized textual data.")
st.subheader("Or enter a url from news stites such as dantri.com and click on the 'Import Url' button to get the summarized.")
st.subheader('This application is powered by Minggz.')
option = st.selectbox(
    'Choosing language: ',
    ('Vietnamese', 'Chinese', 'English','Russian'))
st.write('You selected:', option)
st.write('Paste your copied data or news url here ...')
txt = st.text_area(label='Input', placeholder='Try me', max_chars=5000, height=50)
print(txt)
col1, col2 = st.columns([2,6],gap='large')
with col1:
    summarize_button = st.button(label='Summarize Text')
with col2:
    import_button = st.button(label='Import Url')
summary_length = st.select_slider(label='Summary Length', options=['Extreme Short','Short', 'Medium', 'Long', 'Extreme Long'])
st.write(summary_length)
sum_len_dict = {'Extreme Short':0.2,'Short':0.4, 'Medium':0.6, 'Long':0.8, 'Extreme Long':0.9}
k = sum_len_dict[summary_length]
if option == 'Vietnamese':
    if summarize_button :
        result = vi_sum.summary_doc(txt,k)
        st.write(result)
        print(vi_sum.pretrained)
    if import_button:    
        result = vi_sum.summary_url(txt, k)
        st.write(txt)
        st.write(result)
        print(vi_sum.pretrained)

elif option == 'English':
    if summarize_button :
        result = en_sum.summary_doc(txt,k)
        st.write(result)
        print(en_sum.pretrained)
    if import_button:    
        result = en_sum.summary_url(txt, k)
        st.write(txt)
        st.write(result)
        print(en_sum.pretrained)

elif option == 'Russian':
    if summarize_button :
        result = ru_sum.summary_doc(txt,k)
        st.write(result)
        print(ru_sum.pretrained)
    if import_button:    
        result = ru_sum.summary_url(txt, k)
        st.write(txt)
        st.write(result)
        print(ru_sum.pretrained)
else :
    if summarize_button :
        result = ch_sum.summary_doc(txt,k)
        st.write(result)
        print(ch_sum.pretrained)
    if import_button:    
        result = ch_sum.summary_url(txt, k)
        st.write(txt)
        st.write(result)
        print(ch_sum.pretrained)
    