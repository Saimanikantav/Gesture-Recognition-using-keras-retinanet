
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
from PIL import Image

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
    config = tf.ConfigProto() 
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

#model path
model_path = '/home/sai/RetinanetTutorial/Retinanet-Tutorial/new_weights07_Inference.h5'

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50' )
confidence_cutoff = 0.6

# load label to names mapping for visualization purposes
labels_to_names = {0: 'grab', 1: 'drop'}


#detection from camera

cap = cv2.VideoCapture(0)
counter = 0
sum_time=0
while(True):
    ret, img = cap.read()
    if not ret:
        break
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # preprocess image for network
    image = preprocess_image(bgr)
    
    image,scale = resize_image(bgr)

    # process image
    start = time.time()
    boxes,scores,labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    t = time.time() - start
    print("processing time: ", t)
    boxes /= scale

    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
    # scores are sorted so we can break
        if score < confidence_cutoff:
           break

    #Add boxes and captions
        color = (0, 255, 0)
        thickness = -1
        b = np.array(box).astype(int)
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)
        caption = "{} {:.3f}".format(labels_to_names[label], score)
   
        cv2.putText(img, caption, (b[2], b[3] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0),  2)
        cv2.putText(img, caption, (b[2], b[3] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255),1)
    cv2.imshow('test',img)
    counter=counter+1
    sum_time+=t

    if cv2.waitKey(1) == ord('q'):
       break

cap.release()
cv2.destroyAllWindows()


