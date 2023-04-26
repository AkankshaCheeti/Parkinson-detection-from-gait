import pandas as pd
import os

path = '../dataset_fog_release/dataset/'
count = 0
count1 = 0
# Loop through all files in the directory
for file in os.listdir(path):
    if os.path.isfile(os.path.join(path, file)):
        # Read the file into a dataframe
        df = pd.read_csv(os.path.join(path, file),sep=' ')
        last_col = df.iloc[:, -1].tolist()
        # print(last_col)
        # Check if the last column has the value 2 and print the DataFrame
        if (df.iloc[:, -1] == 2).any():
            count = count +1
        else:
            count1 = count1 +1


print(count)
print(count1)
            
       
      


