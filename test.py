from helper_tools .tciaclient import TCIAClient
from helper_tools.dicom_tools import dcmread_image

from config import config

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import json
import sys
import math
import errno
import pydicom as dicom

from multiprocessing import Pool
import tqdm
import time
from skimage.transform import resize



def dicom_to_numpy(input_tuple):
    view, path, file_num = input_tuple
    im_path = config.TRAIN_DIR / path
    assert os.path.exists(im_path), 'image does not exist'
                    
    image = dcmread_image(fp=im_path, view=view)
    image = image.astype(float)
    image = image/65535.0
    
    scaled_image = resize(image, output_shape=(6,123,95), preserve_range=True,anti_aliasing=True)
    
    save_path = config.LOCAL_TRAIN_IMAGE_DIR / str(file_num)
    np.save(save_path, scaled_image)
    time.sleep(1)
    return save_path


if __name__ == '__main__':


	#directories for data and other files

	PACKAGE_ROOT = config.PACKAGE_ROOT
	TRAIN_DIR = config.TRAIN_DIR
	TEST_DIR = config.TEST_DIR

	df_paths = pd.read_csv(TRAIN_DIR / 'file-paths-train.csv')
	df_paths['id'] = df_paths['classic_path'].apply(lambda x: x.split('/')[3])

	df_target_train = pd.read_csv(TRAIN_DIR / 'BCS-DBT labels-train.csv')

	df_paths = df_paths.merge(df_target_train.drop(['StudyUID','View'],axis=1), on='PatientID')

	images = [TRAIN_DIR / str(r[3]) if os.path.isfile(TRAIN_DIR / str(r[3])) else np.nan for i,r in df_paths.iterrows()]

	images = pd.Series(images)
	mask = ~images.isnull()
	df_subset = df_paths[mask]
	df_subset.head()

	df_sub_target = df_subset.iloc[:,6:]

	input_list = [(r[2],r[3],i) for i,r in df_subset.iterrows()]
	numpy_paths = []

	with Pool(4) as p:
	    numpy_paths =  list(tqdm.tqdm(p.imap(dicom_to_numpy,input_list),total=1000))

	numpy_paths.to_pickle(config.LOCAL_TRAIN_SUBSET)
	df_sub_target.to_pickle(config.LOCAL_TRAIN_TARGET)

