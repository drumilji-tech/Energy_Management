import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
import pandas as pd
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
    data=pd.read_csv("https://raw.githubusercontent.com/drumilji-tech/Energy_Management/main/features_dj.csv")
   
    
    @st.cache(persist=True)
    def split(data):
        
        features = data.drop(columns='score',axis=1)
        targets = data[['score']]
        
        X, X_test, y, y_test = train_test_split(features, targets, test_size = 0.3, random_state = 42)
        return X,X_test,y,y_test 
    
    x_train,x_test,y_train,y_test = split(data)
    
    
    
    
    st.sidebar.subheader('Choose Model')
    Model = st.sidebar.selectbox("Model",('Linear Regression','Decision Tree Regressor','Random Forest Regressor','Support Vector Machine Regressor','Gradient Boosting'))
     
    if Model == "Linear Regression":
        st.sidebar.subheader("Model Hyperparameters")
        max_iter = st.sidebar.slider("Maximum Number of Iterations",100,500,key='max_iter')
        metrics = st.sidebar.selectbox("Which metrics to look?",('Accuracy Score','R2 Score','Mean Squared Error'))
        if st.sidebar.button("Results",key='results'):
            st.subheader("Linear Regression Results")
            Model = LinearRegression()
            Model.fit(x_train,y_train)
            y_pred = Model.predict(x_test)
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
            Model.fit(x_train,y_train)
            y_pred = Model.predict(x_test)
            param_grid = {  'bootstrap': [True], 
                          'max_depth': [5, 10, None], 
                          'max_features': ['auto', 'log2'], 
                          'n_estimators': [5, 6, 7, 8, 9, 10, 11, 12, 13, 15]}
            g_search = GridSearchCV(estimator = Model, param_grid = param_grid, cv = 3, n_jobs = 1, verbose = 0, return_train_score=True)
            g_search.fit(x_train, y_train)
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
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            params = {'max_leaf_nodes': list(range(2, 100)), 
                      'min_samples_split': [2, 3, 4]}
            grid_search_cv = GridSearchCV(DecisionTreeRegressor(random_state=42), params, verbose=1, cv=3)
            grid_search_cv.fit(x_train, y_train)
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
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}
            grid = GridSearchCV(SVR(), param_grid, refit = True, verbose = 3)
            grid.fit(x_train, y_train)
            
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
             model.fit(x_train, y_train)
             parameters = {"n_estimators":[5,50,250,500],"max_depth":[1,3,5,7,9],
                 "learning_rate":[0.01,0.1,1,10,100]}
             cv = GridSearchCV(model,parameters,cv=5)
             st.write('The Best Parametersto be selected to get maximum accuracy for Gradient Boosting Regressor are as follows',+cv.best_params_)
             st.write("Accuracy Score:",accuracy_score(y_test,y_pred).round(4))
             st.write("R2 Value:",r2_score(y_test,y_pred).round(4))
             st.write("Mean Squared Error:",mean_squared_error(y_test,y_pred).round(4))
             
    if st.sidebar.checkbox("Show Raw Data",False):
        st.subheader("New York Consumption Data")
        st.write(data)
        
if __name__ == '__main__':
    main()


        
         
         
    
