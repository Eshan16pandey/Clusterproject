# -*- coding: utf-8 -*-
"""
"""


import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://media.licdn.com/dms/image/C4E12AQENb9Ly2dsucA/article-cover_image-shrink_720_1280/0/1520185057465?e=2147483647&v=beta&t=QbwOqsVcr9awrUnYlZ0rNbQ7Uhs1gPpU3sdT3d6BBjY");
background-size: cover;
}}


[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}

[id="customer-personality-analysis"]{{
color: yellow; 
font-family: "Bookman Old Style", Georgia, serif;
}}

[class="css-10trblm e16nr0p30"]{{
color: yellow; 
font-family: "Bookman Old Style", Georgia, serif;   
}}

</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# loading the saved model
loaded_model = pickle.load(open('logistic_regression_model.pkl'))


# creating a function for Prediction

def cluster_prediction(input_data):
    
    # changing the input_data to numpy array
    input_np = np.asarray(input_data)
  
    # reshape the array as we are predicting for one instance
    data_reshaped = input_np.reshape(1, -1)
    
    df = pd.DataFrame(data_reshaped)
    
    df1 = df.astype(int)
    
    prediction = loaded_model.predict(df1)
    
    if (prediction == 0):
      return 'Good Customer'
    elif (prediction == 1):
        return 'Potentially Good Customer'
    elif (prediction == 2):
        return 'Ordinary Good Customer'
    elif (prediction == 3):
        return 'Elite Customer'
    
    
  
def main():
    

    # giving a title
    st.title('Customer Personality Analysis')
     
    # getting the input data from the user
    st.sidebar.title("Input the Customer Personality Analysis  Factors:")
    
    Income =  st.sidebar.number_input("Customer's yearly household Income:", value=0)
            
    Mntwines = st.sidebar.number_input("Amount spent on WineProducts:",value=0)
        
    Mntmeat =   st.sidebar.number_input("Amount spent on MeatProducts:",value=0) 
        
    Mntfish =  st.sidebar.number_input("Amount spent on FishProducts:",value=0)
    
    Mntsweet =  st.sidebar.number_input("Amount spent on SweetProducts:",value=0)
    
    Mntgold =  st.sidebar.number_input("Amount spent on GoldProducts:",value=0)
        
    Numweb = st.sidebar.number_input("Number of purchases made through the Company’s website:",value=0)
    
    Numcat = st.sidebar.number_input("Number of purchases made using a Catalogue:",value=0)
    
    Numstore = st.sidebar.number_input("Number of purchases made directly in Stores:",value=0)
    
    response = st.sidebar.selectbox("1 if customer accepted the offer in the last campaign, 0 otherwise:",("0","1"))
    
    Tot_spent =  st.sidebar.number_input("Customer's total Spending on Products:", value=0)
    
    Tot_purchase =  st.sidebar.number_input("Customer's total Purchase on Products", value=0)
          
    Childern = st.sidebar.number_input("Total Number of Child in House:",value=0)
    
    famsize = st.sidebar.number_input("Total number of members in the family:",value=0)
    
    isparent = st.sidebar.selectbox("1 if customer is parent, 0 otherwise:",("0","1"))
    
      
    
    # creating a button for Prediction
        
    if st.button('Predict the ClusterID'):
            result = cluster_prediction([Income,Mntwines, Mntmeat,Mntfish,Mntsweet,Mntgold ,Numweb,Numcat ,Numstore,response ,Tot_spent ,Tot_purchase,Childern,famsize,isparent])
            box = st.container()
            with box:
                st.success(result)
            
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                
        

    uploaded_resume = st.file_uploader("Upload your file Here :", type={"csv"})  

    if uploaded_resume is not None:
        df= pd.read_csv(uploaded_resume,index_col=0)
        #st.write(df)
        
        if st.button('Predict the Clusters'):
            
           X = df.iloc[:, :15]
           #handling NA values by dropping the
           df = X.dropna()
            
           scaler = StandardScaler()
           X_scaled = scaler.fit_transform(df)
           
           predict = loaded_model.predict(X_scaled)
           
           X_original = scaler.inverse_transform(X_scaled)

           df['ClusterID'] = predict

           st.write(df)
           
    
           @st.cache_data
           def convert_df(data):
               return data.to_csv().encode('utf-8')

           csv = convert_df(df)

           st.download_button(
               label="Download data as CSV file :arrow_down:",
               data = csv,
               file_name='Clustring.csv',
               mime='csv',)
           
           
           st.subheader("Bar Graph")
           sns.set(palette="pastel")

           # Create the countplot using Seaborn
           plt.figure(figsize=(10, 8))
           sns.countplot(x='ClusterID', data=df)
           plt.ylabel('Count of ClusterID')
           # Display the plot using Streamlit
           st.pyplot(plt)
               
           
            
if __name__ == '__main__':
    main()
    