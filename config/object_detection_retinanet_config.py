# import the necessary packages
import os

# Set the dataset base path here
BASE_PATH = "dataset/sorted"

# build the path to the annotations and input images
ANNOT_PATH = os.path.sep.join([BASE_PATH, 'annotations'])
IMAGES_PATH = os.path.sep.join([BASE_PATH, 'images'])

# degine the training/testing split
# If you have only training dataset then put here TRAIN_TEST_SPLIT = 1
TRAIN_TEST_SPLIT = 0.90

#  build the path to the output training and test .csv files
TRAIN_CSV = os.path.sep.join([BASE_PATH, 'train.csv'])
TEST_CSV = os.path.sep.join([BASE_PATH, 'test.csv'])

# build the path to the input mapping to hyper-classes
CLASS_MAP = os.path.sep.join([BASE_PATH, 'class_map.yml'])

# build the path to the output classes CSV files
CLASSES_CSV = os.path.sep.join([BASE_PATH, 'classes.csv'])

# build the path to the output predictions dir
OUTPUT_DIR = os.path.sep.join([BASE_PATH, 'predictions'])
