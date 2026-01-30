## deploy
##use deploy to create a form for client
import streamlit as st 
import pickle 
from PIL import Image

def main():
    st.title(':red[HEART FAILURE PREDICTION]')
    image=Image.open(r'/Users/wynnthomasthomas/Downloads/PHOTO-2026-01-28-21-05-00.jpg')
    st.image(image,width=600)
    #identify the features
    age=st.text_input('Age','Type here')
    #to add a radio button
    sex=st.radio('sex',['Male','Female'])
    if(sex=='Male'):
        sex=1
    else:
        sex=0
    cp=st.text_input('cp','TYPE HERE')
    trestbps=st.text_input('trestbps','TYPE HERE')
    chol=st.text_input('chol','TYPE HERE')
    fbs=st.text_input('fbs','TYPE HERE')
    restecg=st.text_input('restecg','TYPE HERE')
    thalach=st.text_input('thalach','TYPE HERE')
    exang=st.text_input('exang','TYPE HERE')
    oldpeak=st.text_input('oldpeak','TYPE HERE')
    slope=st.text_input('slope','TYPE HERE')
    ca=st.text_input('ca','TYPE HERE')
    thal=st.text_input('thal','TYPE HERE')

    f=[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
    model1=pickle.load(open('model_knn.sav1','rb'))
    scaler1=pickle.load(open('scaler_knn.save1','rb'))
    pred=st.button('PREDICT')

    if pred:
        prediction=model1.predict(scaler1.transform([f]))
        if prediction==0:
            st.success('not suffering from heart disease')
            st.balloons()
        else:
            st.write('suffering from heart disease')
main()
