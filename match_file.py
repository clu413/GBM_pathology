## match_file.py
# merge clinical information with whole slide image paths 

import pandas as pd
import re
import os
import numpy as np

clinical_data = pd.read_csv('/home/cl427/GBM/clinical/IvyGAPClinicalData_v201806.csv',header = 1)
sub_block =  pd.read_csv('/home/cl427/GBM/clinical/sub_block_details.csv')
clean_df_sub_block = sub_block.loc[:,['donor_id', 'tumor_name','sub_block_id','data_set_id','molecular_subtype','mgmt_methylation','egfr_amplification']]
clean_df_sub_block['sub_block_data_set']=clean_df_sub_block["sub_block_id"].map(str) + '_' +clean_df_sub_block["data_set_id"].map(str) 
clean_df_sub_block['molecular_subtype'] = clean_df_sub_block['molecular_subtype'].replace(to_replace = ', ', value = '_', regex = True)
clinical_data['Tissue ID'] = clinical_data['Tissue ID'].replace(to_replace = ', ', value = '_', regex = True)


file_path = '/n/data2/hms/dbmi/kyu/lab/datasets/IvyGap/HE'
annotation_files = []
files = []
#files = os.listdir(file_path)
with open('/home/cl427/GBM/results/file_list.txt', 'w') as f_list:
    for r, d, f in os.walk(file_path):
        for file in f:
            if 'A.jpg' in file:
                annotation_files.append(os.path.join(r, file))
            
            elif '.jpg' in file:
                files.append(os.path.join(r, file))
                f_list.write(os.path.join(r, file)+'\n')
#print(files)
colnames = list(clean_df_sub_block.columns)
colnames.append('image_path')
print('there are ',len(files),'files')
new_rows_list = []

# find matching file names
for index, row in clean_df_sub_block.iterrows():
    matching = [file for file in files if str(row['sub_block_data_set']) in file]
    if len(matching) != 0:
        for i in range(len(matching)):
            new_row = row.copy()
            new_row['image_path'] = matching[i]
            new_rows_list.append(new_row)
    else: 
        print('no match!')
        #image_path_list.append(np.nan)
        #row['image_path'] =np.nan
        #new_rows_list.append(row)
clean_df_sub_block_long = pd.DataFrame(new_rows_list) 


# extract clinical information
clean_df_clinical_data = clinical_data.loc[:,['Patient ID', 'Tissue ID', 'Allen Institute #', 'Swedish #','History Of Seizures','Age At Initial Diagnosis (in years)','Gender', 'KPS','Mini Mental Status Score','1p19q_deletion','EGFR','PTEN','IDH1']]
df_merge= pd.merge(clean_df_sub_block_long, clean_df_clinical_data, left_on='tumor_name', right_on='Allen Institute #')
df_merge.to_csv('/home/cl427/GBM/results/df_merge_0323.csv')      

