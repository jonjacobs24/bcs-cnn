import pathlib


# PATHS TO FILES

PACKAGE_ROOT = pathlib.Path().resolve()

CNN_DIR = PACKAGE_ROOT / 'cnn_trained'
TRAIN_DIR = PACKAGE_ROOT / 'data/train'
TEST_DIR = PACKAGE_ROOT / 'data/test'

LOCAL_TRAIN_SUBSET = TRAIN_DIR / 'local_training_image_paths.pkl'
LOCAL_TRAIN_TARGET = TRAIN_DIR / 'local_training_target.pkl'

LOCAL_TRAIN_IMAGE_DIR = TRAIN_DIR / 'local_train_images'


#MODEL CONFIG

BATCH_SIZE = 32