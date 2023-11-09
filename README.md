# Retinal Disease Detection using Optical Coherence Tomography and Deep Neural Networks

![Example OCT Image](https://upload.wikimedia.org/wikipedia/commons/9/9f/SD-OCT_Macula_Cross-Section.png)
The healthy macula of a 24 year old male (cross-section view). This image is released to Wikimedia with patient consent. Imaged in-vivo with an Optovue iVue Spectral Domain Optical Coherence Tomographer (SD-OCT) at the office of Drs. Harry Wiessner, Steven Davis, Daniel Wiessner, and Eric Wiessner in Walla Walla, WA, USA.

## Overview

This project aims to develop a deep learning model for the detection of retinal diseases from Optical Coherence Tomography (OCT) images. Optical Coherence Tomography is a non-invasive imaging technique used for high-resolution cross-sectional imaging of the retina. Early detection of retinal diseases such as age-related macular degeneration, diabetic retinopathy, and glaucoma is crucial for timely intervention and treatment.

The project utilizes state-of-the-art deep learning models and transfer learning to create an accurate and robust retinal disease detection system.

## Dataset
The dataset used in this project consists of OCT images categorized into two classes for binary classification:

1. **Abnormal** (combining CNV, DME, and DRUSEN): This class represents retinal diseases, including Choroidal Neovascularization (CNV), Diabetic Macular Edema (DME), and Drusen deposits (DRUSEN).

2. **Normal**: This class represents healthy retinal images.
The dataset is organized into training and test sets, with 6000 images per class in the former and 249 images in the latter. For this binary classification task, the first three classes are merged into the "Abnormal" class, while the "Normal" class remains unchanged. This classification simplifies the task to distinguish between retinal diseases and healthy retinal images.

## Model Architecture
In this project, we explore various pre-trained deep learning architectures, including RESNET-18, RESNET-50, VGG-16, and others, to perform the binary classification of retinal images into "Abnormal" and "Normal" classes. The models are fine-tuned on the binary classification task.

The process involves:

1. **Fine-Tuning**: Pre-trained models are adapted to the specific task of retinal disease detection. This involves updating the top layers of the models while retaining their learned feature extraction capabilities.

2. **Testing and Evaluation**: Each fine-tuned model is rigorously evaluated on the test dataset to assess its performance in terms of accuracy and other relevant metrics.

3. **Model Selection**: The best-performing model, based on rigorous evaluation, will be selected for deployment.

The final choice of the model for deployment will be based on its ability to effectively distinguish between retinal diseases and healthy retinal images. The selection process ensures that the chosen model offers the highest accuracy and robustness for real-world applications.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- NumPy
- Matplotlib
- PIL (Pillow)

The project is still in development stage, more information about the specifics of libraries will be updated after finalizing a model.

## Methodology

1. **Data Collection**: Retinal OCT images were collected from [source](https://data.mendeley.com/datasets/rscbjbr9sj/3).

2. **Data Preprocessing**: Due to computational resource constraints, training and test data were downsampled. Image processing techniques like 'z-score Normalization' and 'Median Filtering' were applied to prepare the dataset for training.

3. **Model Architecture**: All pre-trained feature-extraction layers were frozen and classifications heads were fine-tuned for transfer learning. Final layers of the models were adapted for the binary classification problem.

4. **Model Training**: The deep learning models are trained on the training dataset. Hyperparameters specifics will be updated soon.

5. **Evaluation**: The model's performance is evaluated on the test dataset using appropriate metrics (e.g., accuracy, precision, recall, F1 and ROC-AUC)

6. **Inference**: The trained model can be used for inference on new OCT images to detect retinal diseases.

## Acknowledgments

We would like to acknowledge the source of the OCT dataset used in this project.

Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), “Large Dataset of Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images”, Mendeley Data, V3, doi: 10.17632/rscbjbr9sj.3

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For questions or inquiries, please contact us at shreyasdb99@gmail.com.
