from sklearn.base import BaseEstimator, TransformerMixin
import cv2
import numpy as np
import imutils
from tensorflow.keras.applications.imagenet_utils import preprocess_input

class CropImages(BaseEstimator, TransformerMixin):
    def __init__(self, add_pixels_value=0, target_size=(224, 224)):
        self.add_pixels_value = add_pixels_value
        self.target_size = target_size

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return self.crop_imgs(X, self.add_pixels_value, self.target_size)
    
    def crop_imgs(self, set_name, add_pixels_value, target_size):
        set_new = []
        for img in set_name:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            # Threshold the image, then perform a series of erosions +
            # dilations to remove any small regions of noise
            thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.erode(thresh, None, iterations=2)
            thresh = cv2.dilate(thresh, None, iterations=2)

            # Find contours in thresholded image, then grab the largest one
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            c = max(cnts, key=cv2.contourArea)

            # Find the extreme points
            extLeft = tuple(c[c[:, :, 0].argmin()][0])
            extRight = tuple(c[c[:, :, 0].argmax()][0])
            extTop = tuple(c[c[:, :, 1].argmin()][0])
            extBot = tuple(c[c[:, :, 1].argmax()][0])

            ADD_PIXELS = add_pixels_value
            cropped_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()

            # Resize the cropped image to the target size
            cropped_img_resized = cv2.resize(cropped_img, target_size)

            set_new.append(cropped_img_resized)

        return np.array(set_new)

class PreprocessImages(BaseEstimator, TransformerMixin):
    def __init__(self, img_size=(224, 224)):
        self.img_size = img_size

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return self.preprocess_imgs(X, self.img_size)

    def preprocess_imgs(self, set_name, img_size):
        set_new = []
        for img in set_name:
            img = cv2.resize(
                img,
                dsize=img_size,
                interpolation=cv2.INTER_CUBIC
            )
            set_new.append(preprocess_input(img))
        return np.array(set_new)
from sklearn.pipeline import Pipeline
import joblib

# Create the preprocessing pipeline
preprocessing_pipeline = Pipeline([
    ('crop_images', CropImages(add_pixels_value=0, target_size=(224, 224))),
    ('preprocess_images', PreprocessImages(img_size=(224, 224)))
])


# Save the pipeline
joblib.dump(preprocessing_pipeline, 'preprocessing_pipeline.pkl')
