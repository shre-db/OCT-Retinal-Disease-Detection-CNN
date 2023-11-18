import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
st.set_page_config(page_title="RetinaVision AI", page_icon="icons/braille-solid.svg", initial_sidebar_state='collapsed')

import torch
import torch.nn as nn
from torch.nn.functional import softmax

import torchvision
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

css_model_init = '''
<style>
.model-init {
    font-size: 15px;
    color: #888888;
}
</style>
'''
st.markdown(css_model_init, unsafe_allow_html=True)


st.markdown('<h1 style="text-align: center;"><i class="fa-solid fa-braille"></i> &nbspRetinaVision AI </h1>', unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'><i>Retinal tissue abnormality detection using AI</i></h2>", unsafe_allow_html=True)
st.markdown('')
st.markdown('')
st.markdown('')
st.markdown("""


Welcome to RetinaVision AI

Experience the power of AI in healthcare with RetinaVision AI! Our user-friendly web application empowers you to upload retinal
 optical coherence tomography (OCT) images and receive instant predictions from our state-of-the-art model. Whether you're a healthcare professional or simply curious,
 discover the capabilities of deep learning in detecting abnormalities in retinal tissue. Join us in advancing eye health through
 cutting-edge technology. Upload an image and let the future of vision unfold.
            """)

st.markdown('')
st.markdown('')

st.markdown('<p class="model-init">Initializing the model...</p>', unsafe_allow_html=True)
# Download pre-trained RESNET-18 model
densenet121 = torchvision.models.densenet121()
# Modify final layer
densenet121.classifier = nn.Linear(1024, 2)

st.markdown('<p class="model-init">Loading weights...</p>', unsafe_allow_html=True)
# Load fine-tuned weights
densenet121.load_state_dict(torch.load('models/densenet121-oct-5metrics-v1.pt', map_location=torch.device('cpu')))
densenet121.eval()

st.markdown('<p class="model-init">Model is ready ✔</p>', unsafe_allow_html=True)
st.markdown('')
st.markdown('')

images = st.file_uploader("**CHOOSE AN IMAGE**", type=['jpg'], accept_multiple_files=True)
st.markdown('')
st.markdown('')

if images is not None:
    filenames = [image.name for image in images]
    print(len(filenames))

    transforms = T.Compose([
        T.Resize(256, interpolation=InterpolationMode.BILINEAR),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_tensors = []
    for image in images:
        img = Image.open(image)
        if len(images) == 1:
            st.image(img, use_column_width=True)
        if img.getbands() != 'RGB':
            img = img.convert('RGB')
        transformed_img = transforms(img)
        img_tensors.append(transformed_img)

    inp_tensor = None
    if len(img_tensors) > 0:
        inp_tensor = torch.stack(img_tensors)

    st.markdown('')
    st.markdown('')

    if len(images) > 0:
        if st.button("**Predict**", help="The model evaluates the given images and returns predicted classes and respective confidence scores."):
            def predict(tensor):
                raw = densenet121(tensor)
                y_hat = softmax(raw, dim=1)
                category = torch.argmax(y_hat, dim=1).tolist()
                return category, y_hat.tolist() 
            
            predictions = []
            probability = []
            output = predict(inp_tensor)
            for cat, prob in zip(output[0], output[1]):
                if cat == 0:
                    predictions.append("ABNORMAL")
                    probability.append({round(float(prob[0])*100, 2)})
                elif cat == 1:
                    predictions.append("NORMAL")
                    probability.append({round(float(prob[1])*100, 2)})


            st.markdown('')
            st.markdown('')
            st.markdown('### Results')
            df = pd.DataFrame()
            df['Filename'] = filenames
            df['Predicted Category'] = predictions
            df['Confidence (%)'] = probability
            st.dataframe(df, use_container_width=True, column_order=['Filename', 'Predicted Category', 'Confidence (%)'])
            st.markdown('')


st.markdown('')
st.markdown('')
st.markdown('')
st.markdown('***')

css_copyr = '''
<style>
.footer-text {
    font-size: 15px;
    color: #888888;
    text-align: center;
}
</style>
'''

st.markdown(css_copyr, unsafe_allow_html=True)

st.markdown('<p class="footer-text">Copyright © 2023 &nbsp<i class="fa-solid fa-braille"></i>&nbspRetinaVision AI</p>', unsafe_allow_html=True)
st.markdown("<p class='footer-text'>Contact us at shreyasdb99@gmail.com</p>", unsafe_allow_html=True)
st.markdown('')


css_fa = '''                                                                                                                                                     
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
<style>
.footer-fa {
    font-size: 20px;
    color: #888888;
    text-align: center;
    margin: 0 5px;
    display: inline-block;
}
.footer-icons {
    text-align: center;
}
</style>
<div class="footer-icons">                                                                                                                                                                                                                                                                                               
    <a href="https://github.com/shre-db" target="_blank"><i class="fa-brands fa-github footer-fa"></i></a>                                                                                                                                                                
    <a href="https://www.linkedin.com/in/shreyas-bangera-aa8012271/" target="_blank"><i class="fa-brands fa-linkedin footer-fa"></i></a>                                                                                                                                                                         
    <a href="https://www.instagram.com/shryzium/" target="_blank"><i class="fa-brands fa-instagram footer-fa"></i></a>
</div><br>
<div>
    <p class="footer-text">Version 1.0.0</p>
</div>
'''

st.markdown(css_fa, unsafe_allow_html=True)