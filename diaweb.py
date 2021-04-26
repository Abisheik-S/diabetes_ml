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
    nnn="""<h1 style="color:blue;text-align:center;">Please enter some values , all fields can't be 0 !!</h1>"""
    st.markdown(html_temp,unsafe_allow_html=True)
    BMI = st.text_input('Enter your BMI here (kg/m^2)')
    Gl=st.text_input('Enter your Glucose here (mg/dL)')
    Bp=st.text_input('Enter your Diastolic BloodPressure here (mm Hg)')
    Age=st.text_input('Enter your Age here (years)')
    ST=st.text_input('Enter your Skin Thickness here (mm)')
    preg=st.text_input('Enter your No. of pregnancies here')
    dpf=st.text_input('Enter your Diabetes Pedigree Function  here')
    insulin=st.text_input('Enter your insulin here (mu U/ml)')
    if st.button("predict"):
        a=pd.DataFrame({'Pregnancies':preg, 'Glucose':Gl, 'BloodPressure':Bp, 'SkinThickness':ST, 'Insulin':insulin,
       'BMI':BMI, 'DiabetesPedigreeFunction':dpf, 'Age':Age},index=[0])
        output=model.predict(a)
        if (a['row']== 0).all():
            st.markdown(nnn,unsafe_allow_html=True)
        if(output==0):
            st.markdown(safe,unsafe_allow_html=True)
        else:
            st.markdown(danger,unsafe_allow_html=True)
        
    
if __name__=='__main__':
    main()
