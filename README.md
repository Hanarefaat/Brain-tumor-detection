## README

# Brain Tumor Detection Web Application

This project is a web application for detecting brain tumors from MRI images. It utilizes a trained deep learning model to classify the input MRI images and provides a user-friendly interface for uploading and processing images.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Setup and Installation](#setup-and-installation)
3. [Usage](#usage)
4. [Pipeline Components](#pipeline-components)
5. [Model and Evaluation](#model-and-evaluation)
6. [File Structure](#file-structure)
7. [Detailed Explanation](#detailed-explanation)

## Project Overview
This application uses a deep learning model to detect brain tumors from MRI images. The application is built with Flask and provides an interface for users to upload MRI images, preprocess them, and get predictions about the presence of brain tumors.

## Setup and Installation

### Prerequisites
- Python 3.6+
- Flask
- TensorFlow/Keras
- Joblib
- OpenCV
- Pillow
- Imutils
- Scikit-learn

### Installation Steps

1. **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2. **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Download the dataset:**
    - Download the brain tumor detection dataset from [Kaggle](https://www.kaggle.com/datasets/abhranta/brain-tumor-detection-mri) and place it in your project directory. This dataset contains MRI images of brain tumors categorized into two classes: "No" for images without a tumor and "Yes" for images with a tumor.

4. **Download the trained model:**
    - Download the trained model from [this drive link](https://drive.google.com/file/d/1mqzfVVuAPA8qaHhA1t9CLdLTkxmXv6ln/view?usp=sharing) and place it in your project directory.

5. **Download Bootstrap files:**
    - Download the vendor folder from GitHub and unzip it inside the `static` folder to include the required Bootstrap files for the front end.

### Running the Application

1. **Run the Flask application:**
    ```bash
    python app.py
    ```

2. **Access the application:**
    - Open your web browser and go to `http://127.0.0.1:5000`.

## Usage
1. **Upload an Image:**
    - Navigate to the home page and use the upload functionality to upload an MRI image.

2. **Get Predictions:**
    - The application preprocesses the image, checks if it's a valid MRI image, and then provides a prediction about the presence of a brain tumor along with the probability.

## Pipeline Components

### Preprocessing

The preprocessing pipeline involves several steps to prepare the MRI images for prediction. Two custom transformers are used in the pipeline:

1. **CropImages**: This transformer crops the MRI images to focus on the brain area. It uses computer vision techniques to identify the extreme points in the image and crops the region around these points. The cropped image is then resized to a target size of 224x224 pixels.

2. **PreprocessImages**: This transformer resizes the images and applies necessary preprocessing steps like normalization to prepare the images for input into the deep learning model.

The preprocessing steps ensure that the images are standardized and the model receives data in the expected format. The preprocessing pipeline is saved using Joblib for easy loading during prediction.

### Model

The model used for brain tumor detection is based on the ResNet50 architecture, a deep convolutional neural network pre-trained on ImageNet. The architecture includes additional layers for fine-tuning, such as fully connected layers, batch normalization, dropout for regularization, and a final dense layer with a sigmoid activation function to output the probability of the presence of a brain tumor.

### Training

The training process involves several key steps:
1. **Data Splitting**: The dataset is split into training, validation, and test sets. This helps in training the model, tuning hyperparameters, and evaluating the model's performance on unseen data.

2. **Data Augmentation**: To enhance the model's robustness and generalization, data augmentation techniques are applied. These include rotations, shifts, flips, brightness adjustments, and scaling.

3. **Training Process**: The model is trained using the Adam optimizer and binary cross-entropy loss. Early stopping is used to prevent overfitting, monitoring the validation accuracy to stop training when the performance plateaus.

4. **Evaluation**: The model's performance is evaluated using metrics such as accuracy, precision, recall, F1 score, and ROC AUC score. Confusion matrices are plotted to visualize the model's performance across different classes.

## Model and Evaluation

The evaluation of the model includes:
1. **Accuracy**: The proportion of correctly classified samples.
2. **Precision**: The ability of the model to correctly identify positive cases.
3. **Recall**: The ability of the model to capture all positive cases.
4. **F1 Score**: The harmonic mean of precision and recall.
5. **ROC AUC Score**: The area under the receiver operating characteristic curve, indicating the model's ability to distinguish between classes.
6. **Confusion Matrix**: A matrix showing the true positive, true negative, false positive, and false negative counts, providing insights into the model's classification performance.

The trained model achieves high accuracy and robust performance on the test set, indicating its effectiveness in detecting brain tumors from MRI images.

## Detailed Explanation

### Preprocessing Pipeline

The preprocessing pipeline is defined in `preprocessing.py` and includes two main components: `CropImages` and `PreprocessImages`.

#### CropImages

This custom transformer is responsible for cropping the MRI images to focus on the brain area. The cropping process involves several steps:

1. **Grayscale Conversion**: The image is converted to grayscale to simplify the subsequent processing steps.
2. **Gaussian Blur**: A Gaussian blur is applied to reduce noise and details, which helps in better contour detection.
3. **Thresholding**: The image is thresholded to create a binary image where the brain region is highlighted.
4. **Finding Contours**: Contours are detected in the thresholded image. The largest contour, which corresponds to the brain region, is selected.
5. **Extreme Points Identification**: The extreme points (left, right, top, bottom) of the largest contour are identified.
6. **Cropping**: The image is cropped using the extreme points, and an optional padding (add_pixels_value) can be added.
7. **Resizing**: The cropped image is resized to the target size (224x224 pixels).

#### PreprocessImages

This custom transformer resizes the images to a specified size and applies normalization using the `preprocess_input` function from the Keras applications module. This ensures that the images are in a standardized format before being fed into the neural network.

### Model Architecture

The model is built using the ResNet50 architecture, which is a deep convolutional neural network pre-trained on the ImageNet dataset. The architecture includes several key components:

- **Base Model**: The pre-trained ResNet50 model is used as the base model without the top classification layers. This base model extracts high-level features from the input images.
- **Flatten Layer**: The output of the base model is flattened to create a single long vector. This step converts the 2D feature maps into a 1D feature vector.
- **Batch Normalization**: Batch normalization is applied to stabilize and accelerate the training process by normalizing the output of the previous layer.
- **Dense Layers**: Fully connected layers are added for learning high-level representations. These layers consist of a dense layer with 256 units, followed by batch normalization, ReLU activation, and dropout for regularization.
- **Dropout**: Dropout is used for regularization to prevent overfitting by randomly setting a fraction of input units to zero during training.
- **Output Layer**: A dense layer with a sigmoid activation function outputs the probability of the presence of a brain tumor.

### Training Process

The model is trained using the Adam optimizer and binary cross-entropy loss function. The training process includes the following steps:

1. **Data Splitting**: The dataset is split into training, validation, and test sets. This helps in training the model, tuning hyperparameters, and evaluating the model's performance on unseen data.

2. **Data Augmentation**: Training data is augmented using techniques such as rotations, shifts, flips, brightness adjustments, and scaling. Data augmentation helps in enhancing the model's robustness and generalization ability by artificially increasing the size of the training dataset and introducing variability.

3. **Training Process**: The model is trained using the Adam optimizer with a learning rate of 0.001 and binary cross-entropy loss function. Early stopping is implemented to monitor the validation accuracy and stop training when performance no longer improves, preventing overfitting. The model is trained for a maximum of 30 epochs with a batch size of 32.

4. **Model Evaluation**: The model's performance is evaluated using various metrics such as accuracy, precision, recall, F1 score, and ROC AUC score. Confusion matrices are plotted to visualize the classification performance. The evaluation process involves predicting the class labels for the training, validation, and test sets and comparing them with the true labels to compute the metrics.

The training process is documented in detail in the Jupyter notebook `FINAL.ipynb`.

### Preprocessing Steps in Notebook

The preprocessing steps performed in the notebook for training the model include:

1. **Loading Data**: The MRI images are loaded from the specified directories and split into training, validation,

 and test sets.

2. **Cropping Images**: The `crop_imgs` function is used to crop the images to isolate the brain region. This function performs grayscale conversion, Gaussian blur, thresholding, contour detection, extreme points identification, and cropping.

3. **Image Augmentation**: Data augmentation techniques are applied to the training images using the `ImageDataGenerator` class from Keras. This includes rotations, shifts, flips, brightness adjustments, and scaling.

4. **Preprocessing Images**: The images are resized to the target size (224x224 pixels) and normalized using the `preprocess_input` function.

5. **Saving Preprocessed Images**: The preprocessed images are saved to new directories to be used for training, validation, and testing.

The preprocessing steps in the notebook ensure that the input images are prepared in a similar manner to the images the model was trained on. This is crucial for achieving consistent and accurate predictions during inference.

## File Structure
```
<project-root>
│
├── app.py
├── requirements.txt
├── FINAL.h5
├── preprocessing_pipeline.pkl
├── preprocessing.py
├── FINAL.ipynb
├── templates/
│   ├── first.html
│   ├── login.html
│   ├── chart.html
│   ├── performance.html
│   ├── index.html
│   └── result.html
└── static/
    ├── vendor/
    │   └── bootstrap/
    ├── css/
    │   └── (all CSS files)
    ├── js/
    │   └── main.js
    └── img/
        └── (all images)
```

## Contributing
Feel free to contribute to this project by submitting issues or pull requests.

## License
This project is licensed under the MIT License.

