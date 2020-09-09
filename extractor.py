import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
 
class PoseExtractor(BaseEstimator, TransformerMixin):
   def __init__(self, model_path='./models/pose.tflite'):
       self.model_path = model_path
       self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
       self.interpreter.allocate_tensors()
      
       self.input_details = self.interpreter.get_input_details()
       self.output_details = self.interpreter.get_output_details()
       _, self.input_dim, _, _ = self.input_details[0]['shape']
       _, self.mp_dim, _, self.ky_pt_num = self.output_details[0]['shape']
   def fit(self, x, y=None):
       return self
 
   def extract(self, x):
       # x is list of image paths or numpy arrays
       feature_array = []
       file_path = True if isinstance(x[0], str) else False
       for img in x:
           # Read the image from file path or numpy array and resize it for model
           image = Image.open(row) if file_path else Image.fromarray(img)

image=image.resize((self.input_details[0]['shape'][1],self.input_details[0]['shape'][2]), Image.NEAREST)
image = np.expand_dims(np.asarray(image).astype(self.input_details[0]['dtype'])[:, :, :3], axis=0)
           # Get pose data from the image
           self.interpreter.set_tensor(self.input_details[0]['index'], image)
           self.interpreter.invoke()
           results = self.interpreter.get_tensor(self.output_details[0]['index'])
 
           # Get feature array  from the results
           result = results.reshape(1, self.mp_dim**2, self.ky_pt_num)
           max = np.argmax(result, axis=1)
           coordinates = list(map(lambda x: divmod(x, self.mp_dim), max))
           feature_vector = np.vstack(coordinates).T.reshape(2 * self.ky_pt_num, 1)
           featute_array.append(feature_vector)
 
       return np.array(feature_array).squeeze()

