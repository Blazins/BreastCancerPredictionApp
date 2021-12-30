import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn import preprocessing

#necessary imports

st.title("Breast Cancer Prediction")
st.markdown(
"This app calculates the chances of contracting breast cancer using logistic regression")
## File upload
file_bytes= st.file_uploader("Upload a file", type="csv")

## check whether the file upload has been successful or not
##if successful read the file

if file_bytes is not None:
  def load_data(path):
    data = pd.read_csv(path)
    data = data.drop("id", axis = 1)
    return data
    ## To read non-numeric data transform it into numeric data through encoding
 
  cpd = load_data(file_bytes)
  ### Cleaning data
  def cleaning(data):
    le = preprocessing.LabelEncoder()
    data1 = data
    for i in data1.columns:
      cls = data1[i].dtypes
      if cls == 'object':
        data1[i] = data1[[i]].astype(str).apply(le.fit_transform)
      else:
        data1[i]=data1[i]
    return data1
  Cleaned_Data = cleaning(cpd)
    
  ## Selecting the independent and dependent variables
  st.sidebar.header('Select Output Variable')
  Column_Names = list(cpd.columns)
  Dependent_Var = st.sidebar.selectbox('Dependent Variables', Column_Names)
  
  ## To later alter the results
  Column_Names.remove(Dependent_Var)
  st.sidebar.header('Un-select Variables those are not needed for the analysis')
  Independent_Var = st.sidebar.multiselect('Independent Variables: ', Column_Names, Column_Names)

  #Split test and Train.#Testing is 25% and training is 75%
  X = Cleaned_Data[Independent_Var] ##Defining the X and Y variables
  y = Cleaned_Data[Dependent_Var]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/4, random_state=0)
  ##random-state= 0 for randomness in picking data points
  
  ##predict using logistic regression
  classifier = LogisticRegression()
  classifier.fit(X_train, y_train)
  y_pred = classifier.predict(X_test)
  y_predd = ["Not Affected" if i == 1 else "Affected" for i in y_pred]
  
  cm = confusion_matrix(y_test, y_pred)
  #predict probabilities
  lr_probs_LR = classifier.predict_proba(X_test)
  #keeps probabilities for the positive outcome only
  
  lr_probs_LR = lr_probs_LR[:, 1]
  lr_precision_LR, lr_recall_LR,_ =precision_recall_curve(y_test, lr_probs_LR)
  lr_f1_LR, lr_auc_LR = f1_score(y_test, y_pred), auc(lr_recall_LR, lr_precision_LR)
  
  ## Display accuracy and confusion matrix using write command
  X_test['Prediction'] = y_predd
  X_test['Actual'] = y_test
  st.write('Actual Data Dimension: ' +str(cpd.shape[0]) + 'rows and' + str(cpd.shape[1]) + 'columns.')
  st.dataframe(lr_probs_LR)
  st.write('Test Data Dimension: ' +str(X_test.shape[0]) + 'rows and' + str(X_test.shape[1]) +'columns.')
  st.dataframe(X_test)
  st.write("Confusion Matrix: ")
  st.write(cm)
  st.write('Accuracy: ' +str(accuracy_score(y_test, y_pred)))
  st.write('LogisticRegression :f1 = %.3f auc=%.3f' %(lr_f1_LR, lr_auc_LR))