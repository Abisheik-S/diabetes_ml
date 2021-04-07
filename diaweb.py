# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 16:47:49 2021

@author: abish
"""
import streamlit as st
import pickle
import pandas as pd
model=pickle.load(open('diamodel.pkl','rb'))
    
def main():
    html_temp="""<div style="background-color:#084c46;padding:20px">
    <h1 style="color:white;text-align:center;">Diabetes check</h1>
    </div>"""
    safe="""<h1 style="color:green;text-align:center;">You Are Safe!!</h1> 
    """
    danger="""<h1 style="color:red;text-align:center;">You Are Unsafe !!</h1>"""
    st.markdown(html_temp,unsafe_allow_html=True)
    BMI = st.text_input('Enter your BMI here')
    Gl=st.text_input('Enter your Glucose here')
    Bp=st.text_input('Enter your BloodPressure here')
    Age=st.text_input('Enter your Age here')
    ST=st.text_input('Enter your Skin Thickness here')
    preg=st.text_input('Enter your No. of pregnencies here')
    dpf=st.text_input('Enter your Dpf here')
    insulin=st.text_input('Enter your insulin here')
    if st.button("predict"):
        a=pd.DataFrame({'Pregnancies':preg, 'Glucose':Gl, 'BloodPressure':Bp, 'SkinThickness':ST, 'Insulin':insulin,
       'BMI':BMI, 'DiabetesPedigreeFunction':dpf, 'Age':Age},index=[0])
        output=model.predict(a)
        if(output==0):
            st.markdown(safe,unsafe_allow_html=True)
        else:
            st.markdown(danger,unsafe_allow_html=True)
        
    
if __name__=='__main__':
    main()