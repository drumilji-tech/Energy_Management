import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score,r2_score,mean_squared_error

def main():
    st.title("Smart Water Management Using Data Science")
    st.sidebar.title("Machine Learning and its specifications")
    st.markdown("So, Let's evaluate our model with different Evaluation metrices as the metrices provide us how effective our model is.")
    st.sidebar.markdown("Let\'s do it")
    features=pd.read_csv("https://raw.githubusercontent.com/drumilji-tech/Energy_Management/main/features.csv")
    no_score = features[features['score'].isna()]
    score = features[features['score'].notnull()]
    print(no_score.shape)
    print(score.shape)
    # Separate out the features and targets
    features = score.drop(columns='score')
    targets = pd.DataFrame(score['score'])

    # Replace the inf and -inf with nan (required for later imputation)
    features = features.replace({np.inf: np.nan, -np.inf: np.nan})
    X, X_test, y, y_test = train_test_split(features, targets, test_size = 0.3, random_state = 42)
    print(X.shape)
    print(X_test.shape)
    print(y.shape)
    print(y_test.shape)
    train_features = pd.read_csv('https://raw.githubusercontent.com/drumilji-tech/Energy_Management/main/training_features.csv')
    test_features = pd.read_csv('https://raw.githubusercontent.com/drumilji-tech/Energy_Management/main/testing_features.csv')
    train_labels = pd.read_csv('https://raw.githubusercontent.com/drumilji-tech/Energy_Management/main/training_labels.csv')
    test_labels = pd.read_csv('https://raw.githubusercontent.com/drumilji-tech/Energy_Management/main/testing_labels.csv')
    print('Training Feature Size: ', train_features.shape)
    print('Testing Feature Size:  ', test_features.shape)
    print('Training Labels Size:  ', train_labels.shape)
    print('Testing Labels Size:   ', test_labels.shape)

    imputer = SimpleImputer(strategy='median')

    # Train on the training features
    imputer.fit(train_features)

    # Transform both training data and testing data
    X = imputer.transform(train_features)
    X_test = imputer.transform(test_features)
    print(np.where(~np.isfinite(X)))
    print(np.where(~np.isfinite(X_test))) 
    # Create the scaler object with a range of 0-1
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Fit on the training data
    scaler.fit(X)

    # Transform both the training and testing data
    X = scaler.transform(X)
    X_test = scaler.transform(X_test)
    
    # Convert y to one-dimensional array (vector)
    y = np.array(train_labels).reshape((-1, ))
    y_test = np.array(test_labels).reshape((-1, ))
    
    st.sidebar.subheader('Choose Model')
    Model = st.sidebar.selectbox("Model",('Linear Regression','Decision Tree Regressor','Random Forest Regressor','Support Vector Machine Regressor','Gradient Boosting'))
     
    if Model == "Linear Regression":
        st.sidebar.subheader("Model Hyperparameters")
        max_iter = st.sidebar.slider("Maximum Number of Iterations",100,500,key='max_iter')
        metrics = st.sidebar.selectbox("Which metrics to look?",('Accuracy Score','R2 Score','Mean Squared Error'))
        if st.sidebar.button("Results",key='results'):
            st.subheader("Linear Regression Results")
            Model = LinearRegression()
            Model.fit(X,y)
            y_pred = Model.predict(X_test)
            st.write("Accuracy Score:",accuracy_score(y_test,y_pred).round(4))
            st.write("R2 Value:",r2_score(y_test,y_pred).round(4))
            st.write("Mean Squared Error:",mean_squared_error(y_test,y_pred).round(4))
            st.write("Linear Regression Does not require Hyperparameter Tuning")
    
    if Model == "Random Forest Regressor":
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("The number of trees in the forest",100,5000,step=10,key='n_est')
        max_depth = st.sidebar.number_input("The maximum depth of the tree",1,20,step=1,key='max_depth')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees",('True','False'),key='bootstrap')
        metrics = st.sidebar.selectbox("Which metrics to plot?",('Accuracy Score','R2 Score','Mean Squared Error'),key='1')
        
        if st.sidebar.button("Results",key='results'):
            st.subheader("Random Forest Regression Results")
            Model = RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth,bootstrap=bootstrap)
            Model.fit(X,y)
            y_pred = Model.predict(X_test)
            param_grid = {  'bootstrap': [True], 
                          'max_depth': [5, 10, None], 
                          'max_features': ['auto', 'log2'], 
                          'n_estimators': [5, 6, 7, 8, 9, 10, 11, 12, 13, 15]}
            g_search = GridSearchCV(estimator = Model, param_grid = param_grid, cv = 3, n_jobs = 1, verbose = 0, return_train_score=True)
            g_search.fit(X, y)
            st.write('The Best Parameters for Random Forest are as follows',+g_search.best_params_)
            st.write("Accuracy Score:",accuracy_score(y_test,y_pred).round(4))
            st.write("R2 Value:",r2_score(y_test,y_pred).round(4))
            st.write("Mean Squared Error:",mean_squared_error(y_test,y_pred).round(4))            
            
    if Model == "Decision Tree Regressor":
        st.sidebar.subheader("Model Hyperparameters")
        criterion= st.sidebar.radio('Criterion(measures the quality of split)', ('Gini', 'Entropy'), key='criterion')
        splitter = st.sidebar.radio('Splitter (How to split at each node?)', ('Best','Random'), key='splitter')
        metrics = st.sidebar.selectbox("Which metrics to plot?",('Accuracy Score','R2 Score','Mean Squared Error'),key='1')
        
        if st.sidebar.button("Results",key='results'):
            st.subheader('Decision Tree Results')
            model = DecisionTreeRegressor(criterion=criterion, splitter=splitter)
            model.fit(X,y)
            y_pred = Model.predict(X_test)
            params = {'max_leaf_nodes': list(range(2, 100)), 
                      'min_samples_split': [2, 3, 4]}
            grid_search_cv = GridSearchCV(DecisionTreeRegressor(random_state=42), params, verbose=1, cv=3)
            grid_search_cv.fit(X, y)
            st.write('The Best Parametersto be selected to get maximum accuracy for Random Forest are as follows',+grid_search_cv.best_params_)
            st.write("Accuracy Score:",accuracy_score(y_test,y_pred).round(4))
            st.write("R2 Value:",r2_score(y_test,y_pred).round(4))
            st.write("Mean Squared Error:",mean_squared_error(y_test,y_pred).round(4)) 
            
    if Model == "Support Vector Machine":
        st.sidebar.subheader("Model Hyperparameters")
        kernel= st.sidebar.radio('Type of Kernel to be selected', ('Linear', 'RBF','Ploynomial'), key='kernel')
        C_value = st.sidebar.slider("Select C Value",1,20,key='C_value')
        metrics = st.sidebar.selectbox("Which metrics to plot?",('Accuracy Score','R2 Score','Mean Squared Error'),key='1')
        
        if st.sidebar.button("Results",key='results'):
            st.subheader('SVM Regression Results')
            model = SVR()
            model.fit(X,y)
            y_pred = Model.predict(X_test)
            param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}
            grid = GridSearchCV(SVR(), param_grid, refit = True, verbose = 3)
            grid.fit(X, y)
            
            st.write('The Best Parametersto be selected to get maximum accuracy for SVM are as follows',+grid.best_params_)
            st.write("Accuracy Score:",accuracy_score(y_test,y_pred).round(4))
            st.write("R2 Value:",r2_score(y_test,y_pred).round(4))
            st.write("Mean Squared Error:",mean_squared_error(y_test,y_pred).round(4))
    
    if Model == "Gradient Boosting Regressor":
         st.sidebar.subheader("Model Hyperparameters")
         n_estimators = st.sidebar.number_input("The number of estimators in Gradient Boosting",100,5000,step=10,key='n_est')
         max_depth = st.sidebar.number_input("The maximum depth of Gradient Boosting Regressor",1,20,step=1,key='max_depth')
         learning_rate=st.sidebar_number_input("The learning rate required for Gradient Boosting are",0.01,1000,step=10,key='learning_rate')
         metrics = st.sidebar.selectbox("Which metrics to plot?",('Accuracy Score','R2 Score','Mean Squared Error'),key='1')
         
         if st.sidebar.button("Results",key='results'):
             st.subheader('Gradient Boosting Regression Results')
             model=GradientBoostingRegressor()
             model.fit(X,y)
             y_pred = Model.predict(X_test)
             parameters = {"n_estimators":[5,50,250,500],"max_depth":[1,3,5,7,9],
                 "learning_rate":[0.01,0.1,1,10,100]}
             cv = GridSearchCV(model,parameters,cv=5)
             st.write('The Best Parametersto be selected to get maximum accuracy for Gradient Boosting Regressor are as follows',+cv.best_params_)
             st.write("Accuracy Score:",accuracy_score(y_test,y_pred).round(4))
             st.write("R2 Value:",r2_score(y_test,y_pred).round(4))
             st.write("Mean Squared Error:",mean_squared_error(y_test,y_pred).round(4))
             
    if st.sidebar.checkbox("Show Raw Data",False):
        st.subheader("New York Consumption Data")
        st.write(features)
        
if __name__ == '__main__':
    main()


        
         
         
    
