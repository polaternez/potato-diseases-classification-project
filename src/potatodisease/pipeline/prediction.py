import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
    
    def predict(self):
        # Class names
        potato_disease_list = [
            'Black Scurf', 'Blackleg', 'Common Scab', 'Dry Rot',
            'Healthy Potatoes', 'Miscellaneous', 'Pink Rot'
        ]
        # Load model
        model = load_model(os.path.join("artifacts", "training", "model.h5"))

        # Preprocessing image
        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        # Prediction
        prediction = model.predict(test_image)
        result = potato_disease_list[np.argmax(prediction)]
        print(result)

        return [{"image" : result}]
        