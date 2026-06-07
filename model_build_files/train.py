import os
import joblib
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn import metrics
from sklearn.metrics import(
    mean_squared_error,
    mean_absolute_error,
    r2_score)
from huggingface_hub import login,HfApi,create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow
# Setting the tracking URL for MLflow & defining name of the experiment
if "GITHUB_WORKSPACE" in os.environ:
    base_path = os.environ["GITHUB_WORKSPACE"]
else:
    base_path = os.getcwd()
mlflow.set_tracking_uri(f"file:{os.path.join(base_path,'mlruns')}")
mlflow.set_experiment("bhp-experiment-20")
api = HfApi(token=os.getenv("HF_TOKEN"))
# defining the path to access the splitted datasets
Xtrain_path = "hf://datasets/Lokeshnathy/boston-housing-dataset/Xtrain.csv"
Xtest_path = "hf://datasets/Lokeshnathy/boston-housing-dataset/Xtest.csv"
ytrain_path = "hf://datasets/Lokeshnathy/boston-housing-dataset/ytrain.csv"
ytest_path = "hf://datasets/Lokeshnathy/boston-housing-dataset/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)
numericals = ['CRIM','ZN','NX','RM','DIS','RAD','PTRATIO','LSTAT']
categorical = ['CHAS']
target = 'MEDV'
preprocessor = make_column_transformer(
    (StandardScaler(), numericals),
    (OneHotEncoder(handle_unknown='ignore',drop = 'first'),categorical))
gb_model = GradientBoostingClassifier(random_state=42)
param_grid = {
    'gradientboostingregressor__n_estimators':[100,150,200],
    'gradientboostingregressor__min_samples_leaf':[1,2,3,4,5],
    'gradientboostingregressor__max_features':[0.1,0.2,0.3,0.4,0.5],
    'gradientboostingregressor__max_depth':[3,5,7],
    'gradientboostingregressor__learning_rate':[0.001,0.01,0.1],
    'gradientboostingregressor__min_samples_split':[0,2],
    'gradientboostingregressor__min_impurity_decrease':[0.0,0.001]}
model_pipeline = make_pipeline(preprocessor,gb_model)
with mlflow.start_run():
    grid_search = GridSearchCV(model_pipeline,
                               param_grid,
                               cv=5,
                               n_jobs=-1)
    grid_search.fit(Xtrain,ytrain)
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_neg_mse", mean_score)
    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_
    classification_threshold = 0.45
    y_pred_train = best_model.predict(Xtrain)   
    y_pred_test = best_model.predict(Xtest)
    train_rmse = mean_squared_error(ytrain,y_pred_train)
    test_rmse = mean_squared_error(ytest,y_pred_test)
    train_mae = mean_absolute_error(ytrain,y_pred_train)
    test_mae = mean_absolute_error(ytest,y_pred_test)
    train_r2 = r2_score(ytrain,y_pred_train)
    test_r2 = r2_score(ytest,y_pred_test)
    mlflow.log_metrics({
        "train_RMSE": train_rmse,
        "test_RMSE": test_rmse,
        "train_MAE": train_mae,
        "test_MAE": test_mae,
        "train_R2": train_r2,
        "test_R2": test_r2})
    # saving the best model
    model_path = "best_house_price_predictor_model_v1.joblib"
    joblib.dump(best_model,model_path)
    mlflow.log_artifacts(model_path,artifact_path ="model")
    print(f"Model saved as artifact at: {model_path}")
    repo_id = "Lokeshnathy/best-model-boston-house-price-prediction"
    repo_type = "model"
    try:
        api.repo_info(repo_id=repo_id,repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id,repo_type=repo_type,private=False)
        print(f"Space '{repo_id}' created.")
    # Uploading serialized model to HF Hub
    api.upload_file(
        path_or_fileobj="best_house_price_predictor_model_v1.joblib",
        path_in_repo="best_house_price_predictor_model_v1.joblib",
        repo_id = repo_id,
        repo_type=repo_type)
