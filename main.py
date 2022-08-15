import streamlit as st
import pickle
import pandas as pd
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from PIL import Image
import keras
from keras.models import load_model
#import h5py

cwd = os.getcwd()
model = tf.keras.models.load_model("xray_classfication.h5")

image = Image.open('medical_report.jpg')

st.markdown("<h1 style='text-align: center; color: grey;'>XRAY Classification</h1>", unsafe_allow_html=True)

st.image(image)

st.markdown("<h3 style='text-align: center; color: grey;'>Upload Image</h3>", unsafe_allow_html=True)

image_file = st.file_uploader("", type=["png","jpg","jpeg"])

if image_file is not None:
    img = Image.open(image_file)
    img = np.array(img)
    img = cv2.resize(img,dsize=(150,150))
    img = cv2.cvtColor(np.array(img), cv2.IMREAD_COLOR)
    img = img/255.0
    img = np.array(img).reshape(-1,150,150,3)
    prediction = model.predict(img)
    predicted_val = [int(round(p[0])) for p in prediction]
    pre = predicted_val[0]
    if(pre==0):
        result_image = Image.open('healthy_patient.jpg')
    elif(pre==1):
        result_image = Image.open('unhealthy_patient.jpg')
    st.image(result_image)

# app = Flask(__name__)
# @app.route('/')
# def home():
#     return render_template('index.html')


# @app.route('/predict-xray',methods=["POST"])
# def predict_api():
#     data = request.get_json()
#     print("data {}".format(data))
#     image = request.files['Image']
#     print("image {}".format(image))
#     img = Image.open(image)
#     Image_new = img
#     img = np.array(img)
#     img = cv2.resize(img,(150,150))
#     img = cv2.cvtColor(np.array(img), cv2.IMREAD_COLOR)
#     img = img/255.0
#     #model = pickle.load(open(output_model_path,"rb"))
#     img = np.array(img).reshape(-1,150,150,3)
#     prediction = model.predict(img)
#     predicted_val = [int(round(p[0])) for p in prediction]
#     pre = predicted_val[0]
#     if(pre==0):
#         return "The patient is healthy"
#     elif(pre==1):
#         return "The patient has Pnemonia"


# @app.route('/predict-xray-frontend',methods=["POST"])
# def predict_front_end():
#     data = request.form.get('filename')
#     image = request.files['filename']
#     img = Image.open(image)
#     img = np.array(img)
#     img = cv2.resize(img,dsize=(150,150))
#     img = cv2.cvtColor(np.array(img), cv2.IMREAD_COLOR)
#     img = img/255.0
#     #model = pickle.load(open(output_model_path,"rb"))
#     img = np.array(img).reshape(-1,150,150,3)
#     prediction = model.predict(img)
#     predicted_val = [int(round(p[0])) for p in prediction]
#     pre = predicted_val[0]
#     result = None
#     if(pre==0):
#         #result =  "The patient is healthy"
#         #image = "https://image.shutterstock.com/image-vector/medical-results-vector-illustration-flat-260nw-709891432.jpg"
#         image = "https://image.shutterstock.com/image-photo/satisfied-old-patient-success-young-260nw-381900277.jpg"
#         #color = "#008000"
#     elif(pre==1):
#         # result =  "The patient has Pnemonia"
#         # image = "https://as1.ftcdn.net/jpg/01/71/16/02/500_F_171160247_Zb1dRb071ayvIZDBNqDBoYtdvWtbxbUN.jpg"
#         # color = "#FF0000"
#         image = "https://image.shutterstock.com/image-photo/coronavirus-doctor-visiting-unhealthy-man-260nw-1703608651.jpg"
#     return render_template("index.html",image=image)
#     #return render_template("index.html",image=image,prediction_text=result,color=color)


# if __name__ == "__main__":
#     app.run(port=7000,debug=True)
