
import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

import tensorflow as tf
devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)

model_path = 'snapshots/resnet50_csv_13.h5'    ## replace this with your model path
model = models.load_model(model_path, backbone_name='resnet50')
labels_to_names = {0: 'litter'}                    ## replace with your model labels and its index value

images = os.listdir("dataset/sorted/images/")
image_path = os.path.join("dataset/sorted/images/", np.random.choice(images)) ## replace with input image path
output_path = 'output/detected_image.jpg'   ## replace with output image path

def detection_on_image(image_path):

        image = cv2.imread(image_path)

        draw = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = preprocess_image(image)
        image, scale = resize_image(image)
        out = model.predict_on_batch(np.expand_dims(image, axis=0))
        if len(labels_to_names) > 1:
            boxes, scores, labels = out
        else:
            boxes, scores = out
            labels = [labels_to_names[0] for _ in range(scores.shape[1])]
        boxes /= scale
        for box, score, label in zip(boxes[0], scores[0], labels[0]):

            if score < 0.4:
                break

            color = label_color(label)
            b = box.astype(int)
            draw_box(draw, b, color=color)
            caption = "{} {:.3f}".format(labels_to_names[label], score)
            draw_caption(draw, b, caption)
        detected_img =cv2.cvtColor(draw, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, detected_img)
        cv2.imshow('Detection',detected_img)
        cv2.waitKey(0)
detection_on_image(image_path)