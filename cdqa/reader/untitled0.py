# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 10:11:58 2019

@author: Chacrew
"""
pip install cdqa

import os
import torch
import joblib
from cdqa.reader import BertProcessor, BertQA
from cdqa.utils.download import download_squad


train_processor = BertProcessor(do_lower_case=True, is_training=True, n_jobs=-1)


train_examples, train_features = train_processor.fit_transform(X='KorQuAD_v1.0_train.json')

reader = BertQA(train_batch_size=12,
                learning_rate=3e-5,
                num_train_epochs=2,
                do_lower_case=True,
                output_dir='/save')

reader.fit(X=(train_examples, train_features))

reader.model.to('cpu')
reader.device = torch.device('cpu')

joblib.dump(reader, os.path.join(reader.output_dir, 'bert_qa_korquad_vCPU.joblib'))
