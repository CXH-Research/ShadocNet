import os
from data.dataset import DataLoaderTrain, DataLoaderVal, DataLoaderTest


def get_training_data(img_dir, img_options):
    assert os.path.exists(img_dir)
    return DataLoaderTrain(img_dir, img_options)


def get_validation_data(img_dir, img_options):
    assert os.path.exists(img_dir)
    return DataLoaderVal(img_dir, img_options)


def get_test_data(img_dir, img_options):
    assert os.path.exists(img_dir)
    return DataLoaderTest(img_dir, img_options)
