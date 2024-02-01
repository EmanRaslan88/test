from dash import html,dcc,Dash, Input, Output, callback,State
from jupyter_dash import JupyterDash
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  
  

#Models  
from sklearn.linear_model import LogisticRegression  
from sklearn.model_selection import train_test_split, cross_val_score  

dataframe = pd.read_csv("heart.csv")  
dataframe.dropna(inplace=True)
#Split data into X and y for training features and the target variable  
X=dataframe.drop("num",axis=1)  
y=dataframe["num"]  

X=X[['ca','sex','exang']]
X.shape,y.shape
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)  
#Fit an instance of LR  
clf = LogisticRegression(C=0.20433597178569418,  
                        solver="liblinear")  
  
clf.fit(X_train,y_train) 



app=JupyterDash()
server = app.server

app.layout=html.Div([
dcc.Input(id="input1", type="text"),
dcc.Input(id="input2", type="text"),
dcc.Input(id="input3", type="text"),
html.Button("Submit",id="button"),
html.Div(id="output"),

    
])
@callback(
    Output("output", "children"),
    State("input1", "value"),
    State("input2", "value"),
    State("input3", "value"),
    Input("button","n_clicks")
)
def update_output(input1, input2,input3,button):
    df=pd.DataFrame(np.array([input1, input2,input3]).reshape(1,-1),columns=['ca','sex','exang'])
    result=clf.predict(df)

    return f'result   {result}'

app.run_server()