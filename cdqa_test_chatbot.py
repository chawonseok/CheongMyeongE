# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 15:48:47 2019

@author: Chacrew
"""

import pandas as pd
from ast import literal_eval

from cdqa.utils.filters import filter_paragraphs
from cdqa.utils.download import download_model, download_bnpp_data
from cdqa.pipeline.cdqa_sklearn import QAPipeline
#import time

# Loading data and filtering / preprocessing the documents
df = pd.read_csv('data/bnpp_newsroom_v1.1/jungchat_result.csv',converters={'paragraphs': literal_eval})
#구글드라이브 참조 :https://drive.google.com/drive/folders/1AMu93LP4CEED6mzBGcRB4gQH7OL-i7Iu
#df = filter_paragraphs(df)

# Loading QAPipeline with CPU version of BERT Reader pretrained on SQuAD 1.1
#구글드라이브 참조 :https://drive.google.com/drive/folders/1AMu93LP4CEED6mzBGcRB4gQH7OL-i7Iu
cdqa_pipeline = QAPipeline(reader='models/bert_qa_korquad_vCPU.joblib')

# Fitting the retriever to the list of documents in the dataframe
cdqa_pipeline.fit_retriever(df)
#----------------------------------------------여기까지 실행후 나머지부분 실행해 주세요
#time.sleep(5)
# Sending a question to the pipeline and getting prediction
#query = '지원대상이 누구야?'
#query = '희망두배 청년통장의 지원 금액이 얼마야?'
#query ='청년활동 지원 사업의 지원 금액이 얼마야?'

#query ='청년 허브 운영이 뭐야?'
#prediction = cdqa_pipeline.predict(query)
#
#print('query: {}\n'.format(query))
#print('answer: {}\n'.format(prediction[0]))
#print('title: {}\n'.format(prediction[1]))
#print('paragraph: {}\n'.format(prediction[2]))
print('안녕하세요 서울시 정책챗봇 청명이입니다(나가기:quit)\n무엇을 도와드릴까요?\n')
while True:
    query=input('')
#    print(query+'\n')
    if query=='quit':
        break
#    prediction = cdqa_pipeline.predict(query,n_predictions=3)
    prediction = cdqa_pipeline.predict(query)
    print('paragraph: {}\n'.format(prediction[2]))
