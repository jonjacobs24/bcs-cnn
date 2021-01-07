import pathlib


# PATHS TO FILES

PACKAGE_ROOT = pathlib.Path().resolve()

CNN_DIR = PACKAGE_ROOT / 'cnn_trained'
TRAIN_DIR = PACKAGE_ROOT / 'data/train'
TEST_DIR = PACKAGE_ROOT / 'data/test'

TRAIN_IMAGE_FILE = 'training_images.npz'