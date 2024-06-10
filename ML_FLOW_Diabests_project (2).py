# Databricks notebook source
'''
MLFLOW is an open-source platform for managing end-to-end machine learning lifecycle including exprementation ,reproducibility, and deployment of models
'''
%pip install mlflow

# COMMAND ----------

import pandas as pd # A powrful data manipulation and analysis library

import numpy as np  # A fundamental package for scientific computing with support of arrays and mathematical function

import os #os and sys are modules that provides functionalities to interact with operating system and the python runtime env
import sys
from sklearn  import datasets #A machine learning libirary in python /datasets:utility functions to load datasets 
from itertools import cycle # A function from itertools to iterate over a sequence indefinitely
from sklearn.model_selection import train_test_split # Function to split data into training and testing sets
from sklearn.linear_model import ElasticNet,lasso_path,enet_path # Functions for implementing ElasticNet regression and related path
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score # Metrics to evaluate model performance 
import matplotlib.pyplot as plt # A plotting library to visualize data and model performance
import mlflow  # the main Mlflow package for tracking and managing machine learnning experiemnts
import mlflow.sklearn # specific module within Mlflow for scikit-learn models, providing tools to log save, and load models 

# COMMAND ----------

# MAGIC %pip install mlflow

# COMMAND ----------

#Load data
diabetes=datasets.load_diabetes()
# this function is part of sklearn.datasets modules and is used to load the diabetes dataset
# the diabites dataset is a standard machine learning comunity is used for regression task.it contaIns 442 samples and 10 features ,representing physiological measurments and a target value indicating a quantitative measure of disease progression one year after baseline

x=diabetes.data # extracting features (independent variables) of the dataset 

y=diabetes.target # this attribute contain the target variable (dependent variable) of the dataset. represent the disease progression measure 

np.random.seed(42) 
#  this function is part of the numpy library. the random seed ensures that the random numbers generated in code will be reproducible. This mean that every time you run the code , you will get the same random numbers. which is crucial for ensuring consistency in experiments

# COMMAND ----------

#create pandas dataframe
Y=np.array([y]).transpose()

#  This converts the list y (the target variable) into a NumPy array with an additional dimension, effectively turning it into a 2D array with shape (1, 442).
# .transpose(): This function changes the shape of the array by swapping its axes, resulting in an array of shape (442, 1). Essentially, this turns the target variable into a column vector.

d=np.concatenate((x,Y),axis=1) 

#  This function concatenates the features (x) and the transformed target variable (Y) along the columns (axis 1).
# x has a shape of (442, 10) and Y has a shape of (442, 1). Concatenating these arrays along the columns results in a new array d with shape (442, 11), where the first 10 columns are the features and the 11th column is the target variable.

columns=['age','sex','bmi','bp','s1','s2','s3','s4','s5','s6','progression']

# This line creates a list of column names corresponding to the features and the target variable in the Diabetes dataset.
# age, sex, bmi (body mass index), bp (blood pressure), and s1 to s6 are the 10 features.
# progression is the target variable, indicating the disease progression.

data=pd.DataFrame(d,columns=columns)
# This line creates a list of column names corresponding to the features and the target variable in the Diabetes dataset.
# age, sex, bmi (body mass index), bp (blood pressure), and s1 to s6 are the 10 features.
# progression is the target variable, indicating the disease progression

# COMMAND ----------

data.head()

# COMMAND ----------

'''
This function plot_elastic plots the ElasticNet path for a given dataset and specified l1_ratio. It visualizes how the coefficients of the ElasticNet regression change as a function of the regularization parameter alpha.
x: The feature matrix (independent variables).
y: The target vector (dependent variable).
l1_ratio: The mixing parameter for ElasticNet that controls the balance between L1 and L2 regularization.
eps: A small value added to the alpha values to ensure numerical stability. It defines the minimum value of alpha for the path
global image: Declares a global variable image to store the plot figure, making it accessible outside the function.
enet_path: A function from scikit-learn that computes the coefficients path for ElasticNet regression as a function of alpha.
x: The input data.
y: The target values.
eps: The small value to ensure numerical stability.
l1_ratio: The ElasticNet mixing parameter.
fit_intercept=False: Indicates that the intercept should not be fitted, implying the data should be centered.
fig = plt.figure(): Creates a new figure for plotting.
az = plt.gca(): Gets the current Axes instance on the current figure, creating one if necessary.
colors = cycle(['b', 'r', 'g', 'c', 'k']): Cycles through a set of colors for plotting different coefficient paths.
neg_alpha_ents = -np.log10(alpha_enet): Transforms the alpha values to a logarithmic scale (base 10) for better visualization.
colors = cycle(['b', 'r', 'g', 'c', 'k']): Cycles through a set of colors for plotting different coefficient paths.
neg_alpha_ents = -np.log10(alpha_enet): Transforms the alpha values to a logarithmic scale (base 10) for better visualization.
plt.xlabel('Log Alpha'): Sets the label for the x-axis.
plt.ylabel('Coefficients'): Sets the label for the y-axis.
plt.title(title): Sets the title of the plot, including the l1_ratio.
plt.axis('tight'): Adjusts the plot limits to fit the data tightly.
image = fig: Stores the figure object in the global variable image.
fig.savefig('Elastic_Net-Path.png'): Saves the figure as a PNG file named 'Elastic_Net-Path.png'.
plt.close(fig): Closes the figure to free up memory.
return image: Returns the figure object.

 ----->The plot_elastic function creates and saves a plot that shows how the coefficients of an ElasticNet regression model change with varying regularization parameter alpha. It uses logarithmic scaling for alpha for better visualization, cycles through different colors for plotting each coefficient path, and saves the resulting plot to a file. This plot helps in understanding the effect of regularization on the coefficients, aiding in model selection and tuning.

'''

# COMMAND ----------

def plot_elastic(x,y,l1_ratio):  
    eps=5e-3
    global image
    alpha_enet,coef_enet, _ = enet_path(x,y,eps=eps,l1_ratio=l1_ratio,fit_intercept=False)
    fig=plt.figure()
    az=plt.gca()

    colors=cycle(['b','r','g','c','k'])
    neg_alpha_ents=-np.log10(alpha_enet)

    for coef_e, c in zip(coef_enet,colors):
        l1=plt.plot(neg_alpha_ents,coef_e,c=c,linestyle='--')
    plt.xlabel('Log Alpha')
    plt.ylabel('Coefficients')
    title="ElasticNet path by alpha for l1_ratio= "+str(l1_ratio)
    plt.title(title)
    plt.axis('tight')
    image=fig
    fig.savefig('Elastic_Net-Path.png')
    plt.close(fig)
    return image

# COMMAND ----------

'''
The metric_for_model function calculates three key metrics to evaluate the performance of a regression model: Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared (R²).
actual: The true target values (ground truth).
pred: The predicted target values from the regression model.
mean_squared_error(actual, pred): Computes the Mean Squared Error (MSE) between the actual and predicted values.
---> RMSE: Provides a measure of the average magnitude of the error, giving more weight to larger errors due to the squaring process.

mean_absolute_error(actual, pred): Computes the Mean Absolute Error (MAE) between the actual and predicted values.
---> MAE: The average of the absolute differences between actual and predicted values. It provides a straightforward measure of average error magnitude without considering direction.

R²: Indicates the proportion of variance in the dependent variable that is predictable from the independent variables. It ranges from 0 to 1, with higher values indicating better model performance

'''

# COMMAND ----------

def metric_for_model(actual,pred):
    rmse=np.sqrt(mean_squared_error(actual,pred))
    mae=mean_absolute_error(actual,pred)
    r2=r2_score(actual,pred)
    return rmse,mae,r2

# COMMAND ----------

'''
The train_diabetes_model function is designed to train an ElasticNet regression model on the Diabetes dataset, evaluate its performance, log the results using MLflow, and save the trained model. 
l1_alpha: The alpha value for the ElasticNet model, controlling the strength of regularization.
l1_ratio: The l1_ratio for the ElasticNet model, controlling the balance between L1 and L2 regularization.
train_test_split(data): Splits the data into training and testing subsets.
train_x and test_x: The feature matrices for the training and testing sets, respectively.
train_y and test_y: The target variables for the training and testing sets, respectively.
This section ensures that if no alpha or l1_ratio is provided, default values of 0.05 are used.
Converts the input values to float to ensure compatibility with the ElasticNet model.

mlflow.start_run(): Starts an MLflow run to log parameters, metrics, and artifacts.
ElasticNet(alpha=alpha, l1_ratio=l1_score, random_state=42): Initializes the ElasticNet model with the specified alpha and l1_ratio.
lr.fit(train_x, train_y): Trains the ElasticNet model on the training data.
lr_predict = lr.predict(test_x): Uses the trained model to make predictions on the test data.
(rmse, mae, r2) = metric_for_model(test_y, lr_predict): Calculates evaluation metrics for the model.

Prints the model parameters and evaluation metrics.
Logs the alpha, l1_ratio, RMSE, R², and MAE values as parameters and metrics in MLflow.
Logs the trained model using mlflow.sklearn.log_model.
Saves the model to a specified path using mlflow.sklearn.save_model.

Calls the plot_elastic function to create and save a plot of the ElasticNet paths.
Logs the plot as an artifact in MLflow.
Retrieves and returns the run information.
'''

# COMMAND ----------

def train_diabetes_model(data,l1_alpha,l1_ratio):
    train,test=train_test_split(data)
    train_x=train.drop(['progression'],axis=1)
    test_x=test.drop(['progression'],axis=1)
    train_y=train['progression']
    test_y=test['progression']

    #checks if user have not passed any alpha value
    if float(l1_alpha) is None:
        alpha=0.05
    else:
        alpha=float(l1_alpha)
    if float(l1_ratio) is None:
        l1_score=0.05
    else:
        l1_score=float(l1_ratio)
    
    with mlflow.start_run() as run :
        lr=ElasticNet(alpha=alpha,l1_ratio=l1_score,random_state=42)
        lr.fit(train_x,train_y)
        lr_predict=lr.predict(test_x)
        (rmse,mae,r2)=metric_for_model(test_y,lr_predict)

        print("Elastic Model (alpha=%f ,l1_ratio=%f" % (alpha,l1_score))
        print("RMSE :%s" % rmse)
        print("MAE :%s" % mae)
        print("R2 :%s" % r2)
        mlflow.log_param("alpha",alpha),
        mlflow.log_param("l1_ratio",l1_score),
        mlflow.log_param("RMSE",rmse),
        mlflow.log_param("r2",r2),
        mlflow.log_param("MAE",mae),
        mlflow.sklearn.log_model(lr,"model")
        modelpath="/dbfs/mlflow/test_diabetes/model-%f-%f" %(alpha,l1_score)
        mlflow.sklearn.save_model(lr,modelpath)
        
        image=plot_elastic(x,y,l1_score)
        mlflow.log_artifact('Elastic_Net-Path.png')
        best_run=run.info
        return best_run
        

# COMMAND ----------

# MAGIC %md
# MAGIC #The train_diabetes_model function performs the following steps:
# MAGIC 1. Splits the dataset into training and testing sets.
# MAGIC 2. Handles default values for alpha and l1_ratio if not provided.
# MAGIC 3. Trains an ElasticNet model on the training data.
# MAGIC 4. Evaluates the model on the test data using RMSE, MAE, and R² metrics.
# MAGIC 5. Logs the model parameters, metrics, and artifacts using MLflow.
# MAGIC 6. Saves the trained model and a plot of the ElasticNet paths.
# MAGIC This function provides a comprehensive workflow for training, evaluating, and logging a regression model, facilitating reproducibility and model management.
# MAGIC

# COMMAND ----------

# MAGIC %fs rm -r dbfs:/mlflow/test_diabetes/

# COMMAND ----------

#call the model
train_diabetes_model(data,0.01,0.01)

# COMMAND ----------

display(image)

# COMMAND ----------

train_diabetes_model(data,0.65,0.21)

# COMMAND ----------

display(image)

# COMMAND ----------

best_model=train_diabetes_model(data,0.65,0.21)

# COMMAND ----------

model_name="diabetes_model"   
# This command assigns the string "diabetes_model" to the variable model_name. This variable will presumably be used to name the model when it is logged or registered in MLflow or another model management system.

# COMMAND ----------

'''
This function, print_model_info, is designed to print detailed information about the model(s) passed to it.
The function takes a parameter mod, which is expected to be an iterable (like a list) of models or model versions.
Loop Through Models: It iterates over each item in mod.
Print Model Details: For each model, it prints:
name: The name of the model.
version: The version of the model.
run_id: The run ID associated with the model. This run ID links the model to a specific MLflow run where it was logged.

'''

# COMMAND ----------

def print_model_info(mod):
    for i in mod:
        print("name {}".format(i.name))
        print("version {}".format(i.version))
        print("run_id {}".format(i.run_id))
        print("current_stage {}".format(i.current_stage))


# COMMAND ----------

# MAGIC %md
# MAGIC 1. MLflow Client Initialization: Creates a client to interact with the MLflow tracking server.
# MAGIC 2. Model Registration: Attempts to create a registered model, ignoring any errors if it already exists.
# MAGIC 3. Model Version Creation: Creates a new version of the registered model with the specified artifacts and run ID.
# MAGIC 4. Delay: Pauses the script for 3 seconds to ensure the model version creation process completes.
# MAGIC ---> This code is useful in scenarios where you want to automate the registration and versioning of models in MLflow, ensuring that your models are tracked and versioned systematically.
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

import time
client=mlflow.tracking.MlflowClient() 
try:
    client.create_registered_model(model_name)
except Exception as e:
    pass
model_version=client.create_model_version(model_name,f"{best_model.artifact_uri}/model",best_model.run_id)
time.sleep(3)

# COMMAND ----------

'''
client.get_latest_versions is a method of the MlflowClient class that fetches the latest versions of a specified registered model. Let's detail the parameters and what this line does:
client: The MlflowClient instance created earlier to interact with the MLflow tracking server.
get_latest_versions: This method retrieves the latest versions of the specified model for the given stages.
name=model_name: Specifies the name of the registered model whose versions we want to retrieve. In this case, model_name is "diabetes_model".
stages=["Production"]: This parameter specifies the stages for which we want to get the latest versions. "Production" indicates that we are interested in the versions of the model that are in the "Production" stage.
'''

# COMMAND ----------

model_x=client.get_latest_versions(name=model_name,stages=["Production"])
print_model_info(model_x)

# COMMAND ----------

model_x=client.get_latest_versions(name=model_name)
print_model_info(model_x)

# COMMAND ----------

# transition the model from none stage to stage model
client.transition_model_version_stage(model_name,model_version.version,stage="staging")

# COMMAND ----------

# transition the model from none stage to stage model
client.transition_model_version_stage(model_name,model_version.version,stage="production")

# COMMAND ----------

# transition the model from none stage to stage model
client.transition_model_version_stage(model_name,model_version.version,stage="none")
