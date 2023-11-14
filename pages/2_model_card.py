import streamlit as st
import pandas as pd
st.set_page_config(layout='centered', page_title="RetinaVision AI - Privacy", page_icon='<i class="fa-solid fa-braille"></i>', initial_sidebar_state='collapsed')

st.markdown("### Model Overview")
st.markdown('**Base model**: RESNET-18')
st.markdown("**Head**: Custom head")
st.markdown("**Brief desciption of the model's purpose and use case.**")
st.markdown("""The ResNet-18 model, originally trained on the ImageNet dataset, serves as a powerful starting point for
             medical image classification tasks. In scenarios where computational resources, time constraints, and labeled
             data availability are crucial factors, utilizing a pre-trained model offers several advantages over training from
             scratch.


**1. Time and Resource Savings**: Training a deep neural network from scratch on a medical image dataset can be computationally expensive and time-consuming. Leveraging a pre-trained model allows us to benefit from the knowledge acquired during the training on a diverse and extensive dataset like ImageNet, reducing the overall training time and computational resources required.
Feature Extraction Capability:

**2. Hierarchical Features**: ResNet-18, with its deep architecture, is capable of learning hierarchical features. Lower layers capture low-level features like edges and textures, while higher layers capture more abstract and complex features. This hierarchical feature extraction is advantageous for recognizing patterns in medical images that may have both fine-grained and global characteristics.
Generalization Power:

**3. Capturing Generic Features**: The features learned by ResNet-18 on ImageNet are often generic and transferable to various domains. In medical image classification, where labeled datasets might be limited, the pre-trained model's ability to generalize across different image domains becomes crucial. This aids in achieving reasonable performance even with a smaller medical dataset.
Fine-Tuning Flexibility:

**4. Adapting to Medical Domain**: The ResNet-18 model can be fine-tuned on a smaller medical dataset to specialize its knowledge for the specific task at hand. This fine-tuning process allows the model to adapt its learned features to the intricacies of medical images, improving its performance on the target domain.
Overcoming Data Limitations:

**5. Data Efficiency**: Medical datasets are often smaller than general image datasets like ImageNet. Transfer learning mitigates the challenges associated with limited labeled medical data by initializing the model with weights that already capture valuable image features. This facilitates better model performance, especially when labeled medical data is scarce.
            """)

st.markdown('')
st.markdown("### Model Architecture")
st.markdown("Original RESNET-18 Architecture")
st.markdown("""```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param # 
================================================================
            Conv2d-1         [-1, 64, 112, 112]           9,408
       BatchNorm2d-2         [-1, 64, 112, 112]             128
              ReLU-3         [-1, 64, 112, 112]               0
         MaxPool2d-4           [-1, 64, 56, 56]               0
            Conv2d-5           [-1, 64, 56, 56]          36,864
       BatchNorm2d-6           [-1, 64, 56, 56]             128
              ReLU-7           [-1, 64, 56, 56]               0
            Conv2d-8           [-1, 64, 56, 56]          36,864
       BatchNorm2d-9           [-1, 64, 56, 56]             128
             ReLU-10           [-1, 64, 56, 56]               0
       BasicBlock-11           [-1, 64, 56, 56]               0
           Conv2d-12           [-1, 64, 56, 56]          36,864
      BatchNorm2d-13           [-1, 64, 56, 56]             128
             ReLU-14           [-1, 64, 56, 56]               0
           Conv2d-15           [-1, 64, 56, 56]          36,864
      BatchNorm2d-16           [-1, 64, 56, 56]             128
             ReLU-17           [-1, 64, 56, 56]               0
       BasicBlock-18           [-1, 64, 56, 56]               0
           Conv2d-19          [-1, 128, 28, 28]          73,728
      BatchNorm2d-20          [-1, 128, 28, 28]             256
             ReLU-21          [-1, 128, 28, 28]               0
           Conv2d-22          [-1, 128, 28, 28]         147,456
      BatchNorm2d-23          [-1, 128, 28, 28]             256
           Conv2d-24          [-1, 128, 28, 28]           8,192
      BatchNorm2d-25          [-1, 128, 28, 28]             256
             ReLU-26          [-1, 128, 28, 28]               0
       BasicBlock-27          [-1, 128, 28, 28]               0
           Conv2d-28          [-1, 128, 28, 28]         147,456
      BatchNorm2d-29          [-1, 128, 28, 28]             256
             ReLU-30          [-1, 128, 28, 28]               0
           Conv2d-31          [-1, 128, 28, 28]         147,456
      BatchNorm2d-32          [-1, 128, 28, 28]             256
             ReLU-33          [-1, 128, 28, 28]               0
       BasicBlock-34          [-1, 128, 28, 28]               0
           Conv2d-35          [-1, 256, 14, 14]         294,912
      BatchNorm2d-36          [-1, 256, 14, 14]             512
             ReLU-37          [-1, 256, 14, 14]               0
           Conv2d-38          [-1, 256, 14, 14]         589,824
      BatchNorm2d-39          [-1, 256, 14, 14]             512
           Conv2d-40          [-1, 256, 14, 14]          32,768
      BatchNorm2d-41          [-1, 256, 14, 14]             512
             ReLU-42          [-1, 256, 14, 14]               0
       BasicBlock-43          [-1, 256, 14, 14]               0
           Conv2d-44          [-1, 256, 14, 14]         589,824
      BatchNorm2d-45          [-1, 256, 14, 14]             512
             ReLU-46          [-1, 256, 14, 14]               0
           Conv2d-47          [-1, 256, 14, 14]         589,824
      BatchNorm2d-48          [-1, 256, 14, 14]             512
             ReLU-49          [-1, 256, 14, 14]               0
       BasicBlock-50          [-1, 256, 14, 14]               0
           Conv2d-51            [-1, 512, 7, 7]       1,179,648
      BatchNorm2d-52            [-1, 512, 7, 7]           1,024
             ReLU-53            [-1, 512, 7, 7]               0
           Conv2d-54            [-1, 512, 7, 7]       2,359,296
      BatchNorm2d-55            [-1, 512, 7, 7]           1,024
           Conv2d-56            [-1, 512, 7, 7]         131,072
      BatchNorm2d-57            [-1, 512, 7, 7]           1,024
             ReLU-58            [-1, 512, 7, 7]               0
       BasicBlock-59            [-1, 512, 7, 7]               0
           Conv2d-60            [-1, 512, 7, 7]       2,359,296
      BatchNorm2d-61            [-1, 512, 7, 7]           1,024
             ReLU-62            [-1, 512, 7, 7]               0
           Conv2d-63            [-1, 512, 7, 7]       2,359,296
      BatchNorm2d-64            [-1, 512, 7, 7]           1,024
             ReLU-65            [-1, 512, 7, 7]               0
       BasicBlock-66            [-1, 512, 7, 7]               0
AdaptiveAvgPool2d-67            [-1, 512, 1, 1]               0
           Linear-68                 [-1, 1000]         513,000
================================================================
Total params: 11,689,512
Trainable params: 11,689,512
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.57
Forward/backward pass size (MB): 62.79
Params size (MB): 44.59
Estimated Total Size (MB): 107.96
----------------------------------------------------------------
```""")

st.markdown('')
st.markdown("Modified RESNET-18 Architecture")
st.markdown("""```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 112, 112]           9,408
       BatchNorm2d-2         [-1, 64, 112, 112]             128
              ReLU-3         [-1, 64, 112, 112]               0
         MaxPool2d-4           [-1, 64, 56, 56]               0
            Conv2d-5           [-1, 64, 56, 56]          36,864
       BatchNorm2d-6           [-1, 64, 56, 56]             128
              ReLU-7           [-1, 64, 56, 56]               0
            Conv2d-8           [-1, 64, 56, 56]          36,864
       BatchNorm2d-9           [-1, 64, 56, 56]             128
             ReLU-10           [-1, 64, 56, 56]               0
       BasicBlock-11           [-1, 64, 56, 56]               0
           Conv2d-12           [-1, 64, 56, 56]          36,864
      BatchNorm2d-13           [-1, 64, 56, 56]             128
             ReLU-14           [-1, 64, 56, 56]               0
           Conv2d-15           [-1, 64, 56, 56]          36,864
      BatchNorm2d-16           [-1, 64, 56, 56]             128
             ReLU-17           [-1, 64, 56, 56]               0
       BasicBlock-18           [-1, 64, 56, 56]               0
           Conv2d-19          [-1, 128, 28, 28]          73,728
      BatchNorm2d-20          [-1, 128, 28, 28]             256
             ReLU-21          [-1, 128, 28, 28]               0
           Conv2d-22          [-1, 128, 28, 28]         147,456
      BatchNorm2d-23          [-1, 128, 28, 28]             256
           Conv2d-24          [-1, 128, 28, 28]           8,192
      BatchNorm2d-25          [-1, 128, 28, 28]             256
             ReLU-26          [-1, 128, 28, 28]               0
       BasicBlock-27          [-1, 128, 28, 28]               0
           Conv2d-28          [-1, 128, 28, 28]         147,456
      BatchNorm2d-29          [-1, 128, 28, 28]             256
             ReLU-30          [-1, 128, 28, 28]               0
           Conv2d-31          [-1, 128, 28, 28]         147,456
      BatchNorm2d-32          [-1, 128, 28, 28]             256
             ReLU-33          [-1, 128, 28, 28]               0
       BasicBlock-34          [-1, 128, 28, 28]               0
           Conv2d-35          [-1, 256, 14, 14]         294,912
      BatchNorm2d-36          [-1, 256, 14, 14]             512
             ReLU-37          [-1, 256, 14, 14]               0
           Conv2d-38          [-1, 256, 14, 14]         589,824
      BatchNorm2d-39          [-1, 256, 14, 14]             512
           Conv2d-40          [-1, 256, 14, 14]          32,768
      BatchNorm2d-41          [-1, 256, 14, 14]             512
             ReLU-42          [-1, 256, 14, 14]               0
       BasicBlock-43          [-1, 256, 14, 14]               0
           Conv2d-44          [-1, 256, 14, 14]         589,824
      BatchNorm2d-45          [-1, 256, 14, 14]             512
             ReLU-46          [-1, 256, 14, 14]               0
           Conv2d-47          [-1, 256, 14, 14]         589,824
      BatchNorm2d-48          [-1, 256, 14, 14]             512
             ReLU-49          [-1, 256, 14, 14]               0
       BasicBlock-50          [-1, 256, 14, 14]               0
           Conv2d-51            [-1, 512, 7, 7]       1,179,648
      BatchNorm2d-52            [-1, 512, 7, 7]           1,024
             ReLU-53            [-1, 512, 7, 7]               0
           Conv2d-54            [-1, 512, 7, 7]       2,359,296
      BatchNorm2d-55            [-1, 512, 7, 7]           1,024
           Conv2d-56            [-1, 512, 7, 7]         131,072
      BatchNorm2d-57            [-1, 512, 7, 7]           1,024
             ReLU-58            [-1, 512, 7, 7]               0
       BasicBlock-59            [-1, 512, 7, 7]               0
           Conv2d-60            [-1, 512, 7, 7]       2,359,296
      BatchNorm2d-61            [-1, 512, 7, 7]           1,024
             ReLU-62            [-1, 512, 7, 7]               0
           Conv2d-63            [-1, 512, 7, 7]       2,359,296
      BatchNorm2d-64            [-1, 512, 7, 7]           1,024
             ReLU-65            [-1, 512, 7, 7]               0
       BasicBlock-66            [-1, 512, 7, 7]               0
AdaptiveAvgPool2d-67            [-1, 512, 1, 1]               0
           Linear-68                    [-1, 2]           1,026
================================================================
Total params: 11,177,538
Trainable params: 11,177,538
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.57
Forward/backward pass size (MB): 62.79
Params size (MB): 42.64
Estimated Total Size (MB): 106.00
----------------------------------------------------------------
```""")
st.markdown("""
     All feature extraction layers were frozen and only the classfication head was replaced and fine-tuned for binary classfication.
""")

st.markdown('')
st.markdown("### Training Details")

st.markdown("""
     **Dataset**
     - Training Set (108309 images)
          - ABNORMAL (57169 images)
          - NORMAL (51140 images)
     - Test Set (498 images)
          - ABNORMAL (249 images)
          - NORMAL (249 images)
            
     **Data Preprocessing steps**
     - Grayscale to RGB conversion
     - Resize to (224, 224)
     - Center Crop (224)
     - Convert to PyTorch Tensors
     - Normalize with mean and standard deviation values of [0.485, 0.456, 0.406] and [0.229, 0.224, 0.225] respectively.
            
     **Hyperparameters used during training**
     - Train batch size: 512
     - Test batch size: 249
     - Input image size: (3, 224, 224)
     - Learning Rate: 0.001
     - Optimizer: Adam
     - Loss function: Weighted Cross-Entropy Loss (wCE)
     - Training epochs: 10
     - Number of output units: 2
     - Layer freezing: All feature extraction layers frozen.
""")


st.markdown('')
st.markdown("""
     ### Dataset Information
            
     **Large Dataset of Labeled Optical Coherence Tomography (OCT)**:
     This dataset contains thousands of validated OCT described and analyzed in "Identifying Medical Diagnoses and Treatable
      Diseases by Image-Based Deep Learning". The images are split into a training set and a testing set of independent patients.
      Images are labeled as (disease)-(randomized patient ID)-(image number by this patient) and split into 4 directories:
      CNV, DME, DRUSEN, and NORMAL.
     The folder structure is modified for the binary classification problem, where CNV, DME and DRUSEN are treated as
     ABNORMAL and NORMAL as NORMAL.
     
            
     **Data Source and collection methods**     
     The dataset is contributed by Daniel Kermany, Kang Zhang, Michael Goldbaum at https://data.mendeley.com/datasets/rscbjbr9sj/3
""")
st.markdown('**Data Statistics**')
data = {
     'Medical Condition': ['Choroidal Neovascularization', 'Diabetic Macular Edema', 'Drusen', 'Healthy'],
     'Feature': ['CNV', 'DME', 'DRUSEN', 'NORMAL'],
     'Number of samples (Train, Test)': [(37205, 250), (11348, 250), (8616, 250), (51140, 250)],
}
df = pd.DataFrame(data)
st.dataframe(df, hide_index=True)

st.markdown('**Data Statistics are after folder restructuring and test data downsampling.**')
data = {
     'Medical Condition': ['Abnormal', 'Healthy'],
     'Feature': ['ABNORMAL', 'NORMAL'],
     'Number of samples (Train, Test)': [(51140, 249), (57169, 249)],
}
df = pd.DataFrame(data)
st.dataframe(df, hide_index=True)
st.markdown('Weighted loss function is used to handle class imbalance in the training set.') 


st.markdown('')
st.markdown("""
     ### Model Performance
""")
st.image("performance/performance-resnet18.png")

data = {
     'Metric': ['Accuracy', 'Recall', 'Precision', 'F1', 'ROC-AUC'],
     'Training Set': ['91.88%', '92.98%', '90.19%', '91.52%', '97.38%'],
     'Test Set': ['92.37%', '95.24%', '90.23%', '92.66%', '98.01%'],
}
df = pd.DataFrame(data)
st.dataframe(df, hide_index=True)

st.markdown("""
     - The train and test accuracy, recall, precision, F1 and ROC-AUC scores are all quite high, in the 90%+ range. This indicates that the model has learned the patterns well and is generalizing successfully to new data.
     - The test set scores are very close to the training set scores. This suggests there is minimal overfitting, which is great.
     - The high recall suggests that the model is correctly identifying most of the relevant ABNORMAL cases, at the cost of some precision. This may be desirable for a diagnostic application.            
""")

st.markdown('')
st.markdown("""
     ### Model Input and Output
            
     **1. Description of the input features**
     
     Inputs are optical coherence tomography (OCT) images of retinal tissues having structure and morphology of
      retinal tissues as signal and speckle noise as noise component. RESNET-18 expects 3 channelled, (224 pixel x 224 pixel) sized images for downstream tasks like feature extraction and classification.
     Therefore, images in the form of PyTorch tensors with size (batchsize, 3, 224, 224) are provided as input to the model.
     
     **2. Description and interpretation of the model output**
     
     The final layer of the model produces uninterpretable numbers called 'logits', which can be passed through a softmax activation
     function for interpretation. Since there are two output units in the final layer we get an output tensor of size [batch_size, 2]. An
     example output of raw logits is: `tensor([[-0.9759,  0.7543]], grad_fn=<AddmmBackward0>)`, wherein the numbers are uninterpretable
     until we apply softmax function. The result of applying softmax is `tensor([[0.1506, 0.8494]], grad_fn=<SoftmaxBackward0>)`. The
     numbers 0.1506 and 0.8494 are 'class probabilities' or in layman terms 'confidence' of the model in assigning the input
     image to a particular class. In this example, the model assigns the input image to the second class with 84.94% confidence. Note that
     the sum of the two probabilities is 100%.
""")

st.markdown('')
st.markdown("""          
     ### Model Limitations
            
     **Overall model limitations**

     - As a deep learning model, interpretability is lower compared to some other techniques.
     - Depends on the quality and consistency of the training data labeling and image acquisition process.
            
     **Precision-Recall limitations**

     - High recall but relatively lower precision suggests some over-prediction of abnormal cases. In other words, the model is
      being more inclusive, making an effort to avoid missing any positive instances (minimizing false negatives) even if it means allowing more false positives.
     - Predicted probability thresholds may need adjustment to balance precision and recall for clinical use.
     
""")
st.markdown('')
st.markdown("""
     ### Model Versioning
     
     Version 1.0.0
""")

st.markdown('')
st.markdown("""
     ### Dependencies
     
     **Software requirements** 
     - numpy==1.26.0
     - torch==2.1.0
     - torchvision==0.16.0
     - pillow==10.0.1
     - pandas==2.1.1
     - streamlit==1.28.1
     
     **Hardware requirements**
     - Recommended: GPU
""")

st.markdown('')
st.markdown("""
     ### References
     
     1. Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), “Large Dataset of Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images”, Mendeley Data, V3, doi: 10.17632/rscbjbr9sj.3
""")

st.markdown('')
st.markdown("""
     ### Authors and Contributors
     - Shreyas Bangera
     - Affiliation: Open-Source Project
     - Contact: shreyasdb99@gmail.com
""")

st.markdown('')
st.markdown("""
     ### License
     ```
     MIT License

     Copyright (c) 2023 Shreyas

     Permission is hereby granted, free of charge, to any person obtaining a copy
     of this software and associated documentation files (the "Software"), to deal
     in the Software without restriction, including without limitation the rights
     to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
     copies of the Software, and to permit persons to whom the Software is
     furnished to do so, subject to the following conditions:

     The above copyright notice and this permission notice shall be included in all
     copies or substantial portions of the Software.

     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
     IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
     FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
     LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
     OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
     SOFTWARE.
     ```
""")

st.markdown('')
st.markdown("""
     ### Usage Guidelines
     **Deployment Guidelines:**

     - The model should only be used on OCT images meeting the same quality standards and capture process as the training data. Lower quality images may lead to inaccurate predictions.
     - Prediction outputs should be carefully reviewed and validated by medical professionals before use in any diagnosis or treatment plan. The model cannot be solely relied upon.
     - Monitor model performance with ongoing governance including accuracy, data drift, fairness, and explainability checks. Have a retraining plan in place if performance declines over time.
     - Clinical integration testing is highly recommended prior to full deployment in practice to validate safety and efficacy.
     - To reduce risks related to demographics mismatch, consider retraining or fine-tuning the model on local datasets.
     
     **Appropriate Use Guidelines:**

     - This model is intended only to serve as an assistive screening tool - final diagnosis and treatment decisions must be made by qualified eye care specialists.
     - Outputs should never directly lead to any irreversible treatment or surgical procedures without separate clinical validation.
     - Results should be interpreted in the broader clinical context considering patient history, actual examination and other relevant tests.
     - Use caution when applying model predictions to significantly different patient populations or equipment settings compared to the original training data.
     - Relying solely on model predictions to rule out disease could lead to missed diagnosis - false negatives must be carefully monitored.
     - Adhere to responsible AI practices such as maintaining transparency, evaluating bias/fairness and providing explainability to build appropriate user trust.
     
     By following deployment best practices and usage guidelines focused on clinical validity, safety, and responsibility, this model has the potential to provide valuable assistance improving retinal disease screening and detection.
""")

st.markdown('')
st.markdown("""
     ### Release Date
     Nov 14 2023
""")

st.markdown('')
st.markdown("""
     ### Additional Notes
     - The model was trained using PyTorch and Python with an NVIDIA P100 GPU on a Kaggle environment. Key packages used include NumPy, Matplotlib, Pillowa and Scikit-Learn.
     - The overall model architecture consists of a ResNet18 backbone pre-trained on ImageNet, with a customized classifier head and loss function added.
     - Training was done on a large 108k image dataset. Test set was held out completely from model development.
     - Class imbalance was handled via weighted loss function.
     - The dataset consisted of OCT images sourced from [data.mendeley.com](https://data.mendeley.com/datasets/rscbjbr9sj/3)
     - Regular model monitoring, updated retraining, explainability and bias mitigation procedures should be implemented for production deployment.
     - Future work could explore multi-class classification, localization, ensemble techniques and comparative studies against other retinal image analysis techniques.
""")


st.markdown('')
st.markdown('')
st.markdown("***")

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