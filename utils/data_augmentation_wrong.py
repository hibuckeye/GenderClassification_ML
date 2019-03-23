import numpy as np
from random import shuffle
from .preprocessor import preprocess_input
from .preprocessor import _imread as imread
from .preprocessor import _imresize as imresize
from .preprocessor import to_categorical
import scipy.ndimage as ndi
import cv2

class ImageGenerator(object):
    """ Image generator with saturation, brightness, lighting, contrast,
    horizontal flip and vertical flip transformations. It supports
    bounding boxes coordinates.

    TODO:
        - Finish support for not using bounding_boxes
            - Random crop
            - Test other transformations
    """
    def __init__(self, ground_truth_data, batch_size, image_size,
                train_keys, validation_keys,
                ground_truth_transformer=None,
                path_prefix=None,
                saturation_var=0.5,
                brightness_var=0.5,
                contrast_var=0.5,
                lighting_std=0.5,
                horizontal_flip_probability=0.5,
                vertical_flip_probability=0.5,
                do_random_crop=False,
                grayscale=False,
                zoom_range=[0.75, 1.25],
                translation_factor=.3):

        self.ground_truth_data = ground_truth_data
        self.ground_truth_transformer = ground_truth_transformer
        self.batch_size = batch_size
        self.path_prefix = path_prefix
        self.train_keys = train_keys
        self.validation_keys = validation_keys
        self.image_size = image_size
        self.grayscale = grayscale
        self.color_jitter = []

    def _gray_scale(self, image_array):
        return image_array.dot([0.299, 0.587, 0.114])

    def preprocess_images(self, image_array):
        return preprocess_input(image_array)

    def flow(self, mode='train', model='sequential'):
            while True:
                if mode =='train':
                    shuffle(self.train_keys)
                    keys = self.train_keys
                elif mode == 'val':
                    shuffle(self.validation_keys)
                    keys = self.validation_keys
                else:
                    raise Exception('invalid mode: %s' % mode)

                inputs = []
                targets = []
                for key in keys:
                    image_path = self.path_prefix + key
                    image_array = imread(image_path)
                    image_array = imresize(image_array, self.image_size)

                    num_image_channels = len(image_array.shape)
                    if num_image_channels != 3:
                        continue

                    ground_truth = self.ground_truth_data[key]

                    image_array = image_array.astype('float32')
                    if mode == 'train' or mode == 'demo':
                        if self.ground_truth_transformer != None:
                            image_array, ground_truth = self.transform(
                                                                image_array,
                                                                ground_truth)
                            ground_truth = (
                                self.ground_truth_transformer.assign_boxes(
                                                            ground_truth))
                        else:
                            image_array = self.transform(image_array)[0]

                    if self.grayscale:
                        image_array = cv2.cvtColor(image_array.astype('uint8'),
                                        cv2.COLOR_RGB2GRAY).astype('float32')
                        image_array = np.expand_dims(image_array, -1)

                    inputs.append(image_array)
                    targets.append(ground_truth)
                    if len(targets) == self.batch_size:
                        inputs = np.asarray(inputs)
                        targets = np.asarray(targets)
                        # this will not work for boxes
                        targets = to_categorical(targets)
                        if mode == 'train' or mode == 'val':
                            inputs = self.preprocess_images(inputs)
                            if model == 'functional':
                                yield self._wrap_in_dictionary(inputs, targets)
                            elif model == 'sequential':
                                yield self._warp_in_tuple(inputs, targets)
                        if mode == 'demo':
                            yield self._wrap_in_dictionary(inputs, targets)
                        inputs = []
                        targets = []

    def _wrap_in_dictionary(self, image_array, targets):
        return [{'input_1':image_array},
                {'predictions':targets}]

    def _warp_in_tuple(self, image_array, targets):
        return image_array, targets
