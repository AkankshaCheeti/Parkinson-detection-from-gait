import os
import pandas as pd
import warnings

# ignore warnings
warnings.filterwarnings('ignore')

# define directory path and file format
dir_path = '../dataset2_filtered/'
file_format = 'txt'

# define column names to calculate coefficient of variation
col_names = ['ACC ML [g]', 'ACC AP [g]', 'ACC SI [g]', 'GYR ML [deg/s]', 'GYR AP [deg/s]', 'GYR SI [deg/s]']

# create an empty dataframe to store the results
result_df = pd.DataFrame(columns=['ACC ML [g]', 'ACC AP [g]', 'ACC SI [g]', 'GYR ML [deg/s]', 'GYR AP [deg/s]', 'GYR SI [deg/s]', 'label'])

# iterate through files in directory
for file in os.listdir(dir_path):
    if file.endswith(file_format):
        # read file into pandas dataframe
        df = pd.read_csv(os.path.join(dir_path, file), delimiter='\t')
        # calculate coefficient of variation for selected columns
        cv = df[col_names].std() / df[col_names].mean()
        # add label based on freezing event flag
        if 1 in df['Freezing event [flag]'].values:
            label = 'Pt'
        else:
            label = 'Co'
        # get filename without file extension
        filename = os.path.splitext(file)[0]
        # add result to dataframe
        result_df = result_df.append({'ACC ML [g]': cv['ACC ML [g]'], 
                                      'ACC AP [g]': cv['ACC AP [g]'], 
                                      'ACC SI [g]': cv['ACC SI [g]'], 
                                      'GYR ML [deg/s]': cv['GYR ML [deg/s]'], 
                                      'GYR AP [deg/s]': cv['GYR AP [deg/s]'], 
                                      'GYR SI [deg/s]': cv['GYR SI [deg/s]'], 
                                      'label': label}, ignore_index=True)

# save results to file
res = result_df
result_df.to_csv('../dataset_tl/Parkinson_CV_dataset2.tab', sep='\t', index=False)


from sklearn.preprocessing import PolynomialFeatures

import numpy as np

# select all but the last column as input data
X_without_label = res.iloc[:, :-1].values
label = res.iloc[:,-1].values.reshape(-1, 1)

# create an instance of PolynomialFeatures to transform data to higher dimension
poly = PolynomialFeatures(degree=2, include_bias=False)

# transform data to higher dimension
X_transformed = poly.fit_transform(X_without_label)
X_transformed = X_transformed[:, :16]

# # append label to X_transformed
# X_transformed_with_label = np.concatenate((X_transformed, label), axis=1)

# # print the new shape of X_transformed_with_label
# print(X_transformed_with_label.shape)
# print(X_transformed_with_label)

# X_transformed_with_label.to_csv('../dataset_tl/Parkinson_CV_dataset2_transformed.tab', sep='\t', index=False)

# concatenate X_transformed and label columns horizontally
X_transformed_with_label = np.concatenate([X_transformed, np.array(label).reshape(-1, 1)], axis=1)

# create a dataframe from the numpy array
df = pd.DataFrame(X_transformed_with_label)

# save the dataframe to a tab-separated file
df.to_csv('../dataset_tl/Parkinson_CV_dataset2_transformed.tab', sep='\t', index=False, header=False)
