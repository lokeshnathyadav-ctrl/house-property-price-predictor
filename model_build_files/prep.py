import os
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import login,HfApi
# Loading the dataset
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/Lokeshnathy/boston-housing-dataset/boston.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")
df = df.astype(dtype={'RAD':float,'TAX':float,'CHAS':object})
target = 'MEDV'
numericals = ['CRIM','ZN','NX','RM','DIS','RAD','PTRATIO','LSTAT']
categorical = ['CHAS']
X = df[num_var+cat_var]
y = df[dependent_var]
Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.30,random_state=42)
Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)
split_data = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]
for file_path in dataset_related:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo = file_path.split("/")[-1],
        repo_id = "Lokeshnathy/boston-housing-dataset",
        repo_type="dataset")
