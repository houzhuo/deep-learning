
# coding: utf-8

# In[1]:


from keras.preprocessing.image import img_to_array

class ImageToArrayPreprocessor:
    def __init__(self, dataFormat=None):
        # store the image data format
        self.dataFormat = dataFormat
    
    def preprocessors(self,image):
        # apply the Keras utility function that correctly rearragnes
        # the dimensions of image
        return img_to_array(image, data_format = self.dataFormat)
    
    

