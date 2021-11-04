"""
Predictor interfaces for the Deep Learning challenge.
"""

from typing import List
import numpy as np


# make a prediction for a new image.
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
 

class BaseNet:
    """
    Base class that must be used as base interface to implement 
    the predictor using the model trained by the student.
    """


    def load_model(self, model_path):
        """
        Implement a method to load models given a model path.
        """
        pass

    def predict(
        self, 
        images_a: List, 
        images_b: List, 
        operators: List[str], 
        device: str = 'cpu'
    ) -> List[float]:
        """
        Make a batch prediction considering a mathematical operator 
        using digits from image_a and image_b.
        Instances from iamges_a, images_b, and operators are aligned:
            - images_a[0], images_b[0], operators[0] -> regards the 0-th input instance
        Args: 
            * images_a (List[PIL.Image]): List of RGB PIL Image of any size
            * images_b (List[PIL.Image]): List of RGB PIL Image of any size
            * operators (List[str]): List of mathematical operators from ['+', '-', '*', '/']
                - invalid options must return `None`
            * device: 'cpu' or 'cuda'
        Return: 
            * predicted_number (List[float]): the list of numbers representing the result of the equation from the inputs: 
                [{digit from image_a} {operator} {digit from image_b}]
        """
    # do your magic

    pass 


class RandomModel(BaseNet):
    """This is a dummy random classifier, it is not using the inputs
        it is just an example of the expected inputs and outputs
    """

    def load_model(self, model_path):
        """
        Method responsible for loading the model.
        If you need to download the model, 
        you can download and load it inside this method.
        """
        np.random.seed(42)

    def predict(
        self, images_a, images_b,
        operators, device = 'cpu'
    ) -> List[float]:

        predictions = []
        for image_a, image_b, operator in zip(images_a, images_b, operators):            
            random_prediction = np.random.uniform(-10, 100, size=1)[0]
            predictions.append(random_prediction)
        
        return predictions


class StudentModel(BaseNet):
    """
    TODO: THIS is the class you have to implement:
        load_model: method that loads your best model.
        predict: method that makes batch predictions.
    """
    def operation(i):
        switcher={
                '+':1,
                '-':2,
                '*':3,
                '/':4,
             }
        return switcher.get(i,"Invalid operator")

    def calculator(DigitA,DigitB,operator):
        
        O=StudentModel.operation(operator[1])
        
        if O==1: 
            result=DigitA+DigitB
        elif O==2:
            result=DigitA-DigitB
        elif O==3:
            result=DigitA*DigitB
        elif (O==4) & (DigitB != 0):
            result=DigitA/DigitB
        else:
            result = 'ERROR'

        return result

    # load and prepare the image
    def load_image(filename: str):
	    # load the image
        img = load_img(filename, grayscale=True, target_size=(28, 28))
        #plt.imshow(img)
        #plt.show()
        # convert to array
        img = img_to_array(img)
        # reshape into a single sample with 1 channel
        img = img.reshape(1, 28, 28, 1)
        # prepare pixel data
        img = img.astype('float32')
        img = img / 255.0
        return img

    def load_model(self,model_path: str):
        """
        Load the student's trained model.
        TODO: update the default `model_path` 
              to be the correct path for your best model!
        """
        # load model
        model = load_model(model_path)
        return model 
        
    # TODO:
    def predict(
        images_a, images_b,
        operators, device = 'cuda'
    )-> List[str]:

        """Implement this method to perform predictions 
        given a list of images_a, images_b and operators.
        """

        # load the image
        #img = StudentModel.load_image('deep_equation/resources/trained_model/sample_image.png')
        #img2 = StudentModel.load_image('deep_equation/resources/trained_model/sample_image2.png')
        # load model
        model = load_model('resources/trained_model/final_model.h5')
        # predict the class
        predictions = []

        for image_a, image_b, operator in zip(images_a, images_b, operators):            
            img = StudentModel.load_image(image_a[1:-1])
            img2 = StudentModel.load_image(image_b[1:-1])
            digitA,digitB = model.predict([img,img2])
            Amax=np.argmax(digitA)
            print(Amax)
            Bmax=np.argmax(digitB)
            print(Bmax)
            result=StudentModel.calculator(Amax,Bmax,operator)
            predictions.append(result)

        
        return predictions
 
#Run


with open('resources/data/inputs.txt') as f:
    lines = [line.rstrip() for line in f]

images_a=[]
images_b=[]
operators=[]

for i in range(len(lines)):
    x = lines[i]
    result = x.split(',')
    
    images_a.append(result[0])
    images_b.append(result[1])
    operators.append(result[2])

predictions=StudentModel.predict(images_a,images_b,operators,device='cuda')
print(predictions)