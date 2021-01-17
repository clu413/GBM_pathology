## train_vs_test_by_pt.py

# stores information about each tile into corresponding set/patient annotations
# e.g. coordinates, tile label, patient id

# usage:
# python3 train_vs_test_by_pt.py patient_id train/val/test

# Chenyue Lu
# 2020.03.26

import pandas as pd
import random
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from datetime import datetime
import sys

patient = sys.argv[1]
set_type = sys.argv[2]

full_df = pd.read_csv('/home/cl427/GBM/results/df_merge_0323.csv')

print(patient)
print(set_type)
print('Current time:'+datetime.now().strftime("%H:%M:%S"))
sub_block_data_set_df = full_df[full_df['donor_id']==int(patient)]['sub_block_data_set']
text = []
df = pd.DataFrame()
for sub_block_data_set in sub_block_data_set_df:
    #print(sub_block_data_set)
    for text_file in  list(glob.glob('/n/data2/hms/dbmi/kyu/lab/cl427/segmentation/'+sub_block_data_set+'/*/*.txt')):
        #print(text_file)
        text.append([str(patient),sub_block_data_set,text_file])
        annotation_df = pd.read_csv(text_file, delimiter='\t')
        annotation_df.columns = ['x_coord', 'y_coord', 'label']
        annotation_df['path'] = text_file.replace('annotation_R.txt','seg_')+annotation_df['x_coord'].apply(str) +'_'+ annotation_df['y_coord'].apply(str)+'.jpg'
        annotation_df['patient']=str(patient)
        annotation_df['sub_block_data_set']=sub_block_data_set
        annotation_df['annotation_file']=text_file
        df=df.append(annotation_df)
df.to_csv('/home/cl427/GBM/results/train_vs_test_by_pt/'+set_type+'/'+patient+'_df.txt')
