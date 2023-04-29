import numpy as np
import os
import pandas as pd
import csv

def calculate_cv(df):
    cv = df.std() / df.mean()
    return cv.tolist()



def transform_into_frequency_domain(data):
    # Compute the Fast Fourier Transform (FFT)
    fft_data = np.fft.fft(data)
    
    # Compute the magnitudes of the FFT coefficients
    magnitudes = np.abs(fft_data)
    
    # Return the first n/2 magnitudes (to remove duplicate information due to symmetry)
    n = len(data)
    # print(magnitudes[:n//2][0])
    return list(magnitudes[:n//2][0])


def find_label(filename):
    label = filename.split('_')[0][2:4]
    return label


import csv
import os
import pandas as pd

result_directory ='../dataset_tl/'
directory_path = '../dataset/'
cols=['time', 'l1','l2','l3','l4','l5','l6','l7','l8','r1','r2','r3','r4','r5','r6','r7','r8','total_l','total_r']
filtered_cols = ['l1','l2','l3','l4','l5','l6','l7','l8','r1','r2','r3','r4','r5','r6','r7','r8']

with open(result_directory+'Parkinson_FD.tab', 'w', newline='') as file:
    writer = csv.writer(file, delimiter='\t')
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            df = pd.read_csv(file_path, delimiter='\t', header=None, names=cols)
            df_new = df[filtered_cols]
            if(len(df_new) > 8000):
                filtered_df = df_new.iloc[:8000, :]
                res =transform_into_frequency_domain(filtered_df)
                res.append(find_label(filename))
                writer.writerow(res)




import csv
import os
import pandas as pd

result_directory ='../dataset_tl/'
directory_path = '../dataset/'
cols=['time', 'l1','l2','l3','l4','l5','l6','l7','l8','r1','r2','r3','r4','r5','r6','r7','r8','total_l','total_r']
filtered_cols = ['l1','l2','l3','l4','l5','l6','l7','l8','r1','r2','r3','r4','r5','r6','r7','r8']

with open(result_directory+'Parkinson_CV.tab', 'w', newline='') as file:
    writer = csv.writer(file, delimiter='\t')
    header = filtered_cols + ['label']
    writer.writerow(header)
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            df = pd.read_csv(file_path, delimiter='\t', header=None, names=cols)
            df_new = df[filtered_cols]
            cv = calculate_cv(df_new)
            writer.writerow(list(cv) + [find_label(filename)])