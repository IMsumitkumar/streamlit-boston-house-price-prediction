# Import necessary libraries
import os
import pickle
import streamlit as st
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt     
import seaborn as sns 
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# changing page main title and main icon(logo)
PAGE_CONFIG = {"page_title":"Boston House Price prediction", "page_icon":":house:", "layout":"centered"}
st.set_page_config(**PAGE_CONFIG)   

st.sidebar.text("Created on Thu, Dec 17 2020")
st.sidebar.markdown("**@author:Sumit Kumar** :monkey_face:")
st.sidebar.markdown("[My Github](https://github.com/IMsumitkumar) :penguin:")
st.sidebar.markdown("[findingdata.ml](https://www.findingdata.ml/) :spider_web:")
st.sidebar.markdown("coded with :heart:")

# sidebar header
st.sidebar.subheader("Bsoton House's")


# loading dataset and storing it into local caching for faster resposes
@st.cache
def load_dataset():
    # fetching raw data direct from scikit learn
    boston = load_boston()
    # fetching independent features 
    bos = pd.DataFrame(boston.data)
    # fetching dependent features
    target = pd.DataFrame(boston.target)
    # rename the dataset columns with actual names, can be obtained by data.columns
    bos = bos.rename(columns={0:'CRIM', 1:'ZN', 2:'INDUS', 3:'CHAS', 4:'NOX', 5:'RM', 6:'AGE', 7:'DIS', 8:'RAD', 9:'TAX', 10:'PTRATIO',
                            11:'B', 12:'LSTAT'})
    target = target.rename(columns={0:'MEDV'})

    # concatinating dependent and independent features
    data = pd.concat([bos, target], axis=1)
    return data

# sidebar : choose analysis or prediction page
option = st.sidebar.selectbox(
    'analysis or prediction?',
     ("Please Select here", "Boston House Price Prediction", "Boston House Price"))

if option == "Please Select here":
    st.title("Boston House Prices")
    st.text("Regression predictive modeling machine learning problem from end-to-end Python")

    st.title("")
    st.image("https://i.imgur.com/wwRt3Xf.jpg", width=700)


# choosed prediction page? OK!
if option == 'Boston House Price Prediction':
    # adding image
    st.image("https://i.imgur.com/dVV04u3.jpg", width=700)
    # add title
    st.title("Boston House Price Prediction")

    # initialize dataset from local cache
    data = load_dataset()   

    # show data if checked
    show_data = st.sidebar.checkbox("Show data")
    if show_data:
        st.subheader("Raw Data")
        st.dataframe(data)

    # dividing container into 2 columns
    left_column, right_column = st.beta_columns(2)

    # distributing input fields in both te columns
    rm = left_column.number_input("RM")
    zn = right_column.number_input("ZN")
    lstat = left_column.number_input("LSTAT")
    tax = right_column.number_input("TAX")
    ptratio = left_column.number_input("PTRATIO")
    age = right_column.number_input("AGE")
    dis = left_column.number_input("DIS")
    nox = right_column.number_input("NOX")

    # dependent and independent features
    X = data[['RM','ZN','LSTAT','TAX','PTRATIO','AGE','DIS','NOX']]
    y = data['MEDV']

    # scaling the dataset
    scale = StandardScaler()
    X_scaled = scale.fit_transform(X[3:4])

    # loading the trained pickle file
    loaded_model = pickle.load(open('final_linear_model.pickle', 'rb')) 

    # ppredict using pretrained model -- linear regression
    if st.button("Predict"):
        scaled = StandardScaler()
        scaled.fit_transform([[rm, zn, lstat, tax, ptratio, age, dis, nox]])

        # showing a message using a success badge
        st.success(loaded_model.predict(scale.transform([[rm, zn, lstat, tax, ptratio, age, dis, nox]])))
        

# analysis choosed? OK!
elif option == "Boston House Price":
    # adding title of the page
    st.title("Boston House Price dataset")
    # adding a warning
    st.warning("**Check Atmost Two Checkbox at a time for faster Response!**")

    # adding problem statement box as a sidebar context
    st.sidebar.subheader("problem Statement")
    st.sidebar.success("The dataset used in this project comes from the UCI Machine Learning Repository. This data was collected in 1978 and each of the 506 entries represents aggregate information about 14 features of homes from various suburbs located in Boston.Dataset is having 0 null values and task is to predict the price of the house -MEDV- based on other revelent features.")
    data = load_dataset()

    # telling the shape of the dataset
    st.markdown("Dataset has **{}** Records and **{}** Features".format(data.shape[0], data.shape[1]))
    # no missig values COOL!
    # data.isna().sum()
    st.write("No special and missing values")

    # show the raw data if checked
    show_data = st.checkbox("Show data")
    if show_data:
        st.subheader("Raw Data")
        st.dataframe(data)
        
    # show features description if checked
    if st.checkbox("Show Features Description"):
        # Sub heading
        st.subheader("Features Description")
        # can be used markdown here in context
        st.markdown("""
        `CRIM` per capita: crime rate by town

        `ZN`: proportion of residential land zoned for lots over 25,000 sq.ft.

        `INDUS`: proportion of non-retail business acres per town

        `CHAS`: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)

        `NOX`: nitric oxides concentration (parts per 10 million)

        `RM`: average number of rooms per dwelling

        `AGE`: proportion of owner-occupied units built prior to 1940

        `DIS`: weighted distances to five Boston employment centres

        `RAD`: index of accessibility to radial highways

        `TAX`: full-value property-tax rate per 10,000usd

        `PTRATIO`: pupil-teacher ratio by town

        `B` : 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town

        `LSTAT`: % lower status of the population

        `MEDV `--> our resident value target
        """)


    # show distributio of dependent feature if checked
    if st.checkbox("Dependent Feature distribution and Q-Q plot"):
        st.subheader("Dependent Feature distribution and Q-Q plot")
        # Dividing the container into two columns
        left_column, right_column = st.beta_columns(2)
        
        # distribution and probability plot
        fig = plt.figure()
        sns.distplot(data['MEDV'], fit=stats.norm)
        left_column.pyplot(fig)
        fig = plt.figure()
        res = stats.probplot(data['MEDV'], plot=plt)
        right_column.pyplot(fig)
    
    # Show distributions of all the independent features with boxplot
    if st.checkbox("Data Distributions"):
        def box_dis(data, col, ax):
            sns.set(style="darkgrid", palette="pastel")

            # calculating mean and median and mode
            mean = data[col].mean()
            median = data[col].median()
            mode = data[col].mode()[0]
            
            # dividing the subplots
            f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw= {"height_ratios": (0.2, 1)})
            
            # boxplot and its 
            sns.boxplot(data[col], ax=ax_box)
            ax_box.axvline(mean, color='r', linestyle='--')
            ax_box.axvline(median, color='g', linestyle='-')
            ax_box.axvline(mode, color='b', linestyle='-')
            ax_box.set_ylabel(col, fontsize=15)
            
            # distplot and it's
            sns.distplot(data[col], ax=ax_hist)
            ax_hist.axvline(mean, color='r', linestyle='--')
            ax_hist.axvline(median, color='g', linestyle='-')
            ax_hist.axvline(mode, color='b', linestyle='-')
            
            plt.legend({'mean':mean, 'Median':median, 'Mode':mode})
            ax_box.set(xlabel='')
            ax.pyplot(f)

        st.subheader("Data distributions : let'get an idea of distrbutions and outliers if any!")
        one, two = st.beta_columns(2)
        three, four = st.beta_columns(2)
        five, six = st.beta_columns(2)
        seven, eight = st.beta_columns(2)
        nine, ten = st.beta_columns(2)
        eleven, twelve = st.beta_columns(2)

        box_dis(data=data, col='CRIM', ax=one)
        box_dis(data=data, col='ZN', ax=two)
        box_dis(data=data, col='INDUS', ax=three)
        box_dis(data=data, col='NOX', ax=four)
        box_dis(data=data, col='RM', ax=five)
        box_dis(data=data, col='AGE', ax=six)
        box_dis(data=data, col='DIS', ax=seven)
        box_dis(data=data, col='RAD', ax=eight)
        box_dis(data=data, col='TAX', ax=nine)
        box_dis(data=data, col='PTRATIO', ax=ten)
        box_dis(data=data, col='B', ax=eleven)
        box_dis(data=data, col='LSTAT', ax=twelve)

    # show heat map of all the features if checked
    if st.checkbox("A vey useful : Heat map"):
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        corr_data = data
        corr = corr_data.corr(method="spearman")
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        f, ax = plt.subplots(figsize=(15, 15))
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot = True)
        st.pyplot(f)

    
    # show VIF of selected features : Vif more than 5 is bad ; if checked
    if st.checkbox("Variance inflection factor VIF for features"):
        new_data = data.drop(columns=['MEDV'], axis=1)
        features_selected = st.multiselect("Select best features", new_data.columns)
        
        if len(features_selected) >= 2:

            X = data[features_selected] 
            scale = StandardScaler()
            X_scaled = scale.fit_transform(X)

            variables = X_scaled
            vif = pd.DataFrame()
        
            vif["VIF"] = [variance_inflation_factor(variables, i) for i in range(variables.shape[1])]
            vif["Features"] = X.columns

            vif

            for i in range(len(vif["VIF"])):
                if i >= 5:
                    st.error("{} is not a good feature".format(vif["Features"][i]))
        else:
            st.error("Select at least two")

    # finally, train the model
    if st.checkbox("Train the model", value=True):
        # data without dependent feature
        new_data = data.drop(columns=['MEDV'], axis=1)
        # divide container into two columns
        left, right = st.beta_columns(2)
        # multiselect : select multiple features
        features_selected = left.multiselect("Select best features for training", new_data.columns)
        # select the model on which we want to train
        model_selected = right.selectbox('Choose Linear model',("Linear Regression",))
        # if features selected
        if len(features_selected) >= 1:
            # start training process if checked
            if st.button("Train the model"):

                std = st.text("Standardizing the data...")
                std.text("Data is standardized!")
                splitting = st.text("splitting dataset into training and testing datasets...")
                
                # scaling and splitting data 
                X = data[features_selected]
                scale = StandardScaler()
                X_scaled = scale.fit_transform(X)
                y = data["MEDV"]
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=1)

                splitting.text("Data is splitted into training and testing!")
                # if selected model is linear regression then
                if model_selected == "Linear Regression":
                    model_int = st.text("Model initialized!")

                    # initialization of model
                    lr = LinearRegression()
                    model_int.text("{} model is initialized".format(model_selected))
                    fit_model = st.text("Training dataset is fitting in model...")
                    lr.fit(X_train, y_train)
                    fit_model.text("Training Dataset is fitted!")

                    # printing intercept and slope of best fit line 
                    st.code("Slope is {}".format(lr.intercept_))
                    for i, col in enumerate(features_selected):
                        st.text('The Coefficient of  {} is {}'.format(col, round(lr.coef_[i],3)))

                    st.text("predicting on test dataset...")
                    preds = lr.predict(X_test)
                    score = r2_score(y_test, preds)
                    st.success("Score is {} %".format(round(score, 2)*100))

                    # fuction to calculate adjusted r squared 
                    @st.cache
                    def adj_r2(x,y):
                        r2 = lr.score(x,y)
                        n = x.shape[0]
                        p = x.shape[1]
                        adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
                        return adjusted_r2

                    # calculating adjusted r squared 
                    ad_r2 = adj_r2(X_test, y_test)
                    st.success("Adjusted R2 score is {}".format(round(ad_r2, 2)))

                    # plotting the best fit line 
                    st.subheader("Visualize the best fit line")
                    sns.set(style="darkgrid", palette="pastel")
                    f = plt.figure(figsize=(10,5))
                    plt.scatter(y_test, preds, color='black')
                    plt.plot([y.min(), y.max()], [y.min(), y.max()], c='red', lw=2)
                    st.pyplot(f)




            








