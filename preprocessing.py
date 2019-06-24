import os
import cv2
import numpy as np
from scipy import ndimage

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(ROOT_DIR, 'images')


class TanTriggsPreprocessing:
    def __init__(self, alpha=0.1, tau=10.0, gamma=0.2, sigma0=1.0, sigma1=2.0):
        self._alpha = float(alpha)
        self._tau = float(tau)
        self._gamma = float(gamma)
        self._sigma0 = float(sigma0)
        self._sigma1 = float(sigma1)

    def compute(self, X):
        Xp = []
        for xi in X:
            Xp.append(self.extract(xi))
        return np.array(Xp, dtype='uint8')

    def extract(self, X):
        X = np.array(X, dtype=np.float32)
        X = np.power(X, self._gamma)
        X = np.asarray(ndimage.gaussian_filter(X, self._sigma1) -
                       ndimage.gaussian_filter(X, self._sigma0))
        X = X / \
            np.power(np.mean(np.power(np.abs(X), self._alpha)), 1.0/self._alpha)
        X = X / np.power(np.mean(np.power(np.minimum(np.abs(X),
                                                     self._tau), self._alpha)), 1.0/self._alpha)
        X = self._tau*np.tanh(X/self._tau)
        return X


def test_preprocessing(img_path):
    img = cv2.imread(img_path)
    preprocessing_algo = TanTriggsPreprocessing()
    processed = np.array(preprocessing_algo.compute(img))

    cv2.imshow('show processed', processed)
    cv2.waitKey(10000)


if __name__ == '__main__':
    image_path = os.path.join(images_dir, 'emilia-clarke', '1.jpg')
    test_preprocessing(image_path)
