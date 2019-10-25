import tensorflow as tf
from tensorflow.python import keras
from models.keras_mobilenet import MobileNet
from models.keras_mobilenet_v2 import MobileNetV2
import numpy as np


def make_mobilenetv1_base_weights():
    inputs = keras.Input((112, 160, 3))

    from tensorflow.python.keras.applications import MobileNet as o_moblie
    for alpha in [.25, .5, .75, 1.]:
        om = o_moblie(input_tensor=inputs, alpha=alpha, include_top=False)  # type:keras.Model
        oweights = om.get_weights()
        nm = MobileNet(input_tensor=inputs, alpha=alpha)
        nweights = nm.get_weights()

        for i in range(len(nweights)):
            nweights[i] = oweights[i][tuple([slice(0, s) for s in nweights[i].shape])]

        nm.set_weights(nweights)

        keras.models.save_model(nm, f'data/mobilenet_v1_base_{int(alpha*10)}.h5')


def make_mobilenetv2_base_weights():
    inputs = keras.Input((112, 160, 3))
    from tensorflow.python.keras.applications import MobileNetV2 as o_moblie
    for alpha in [.35, .5, .75, 1.]:
        om = o_moblie(input_tensor=inputs, alpha=alpha, include_top=False)  # type:keras.Model
        oweights = om.get_weights()
        nm = MobileNetV2(input_tensor=inputs, alpha=alpha, weights=None, include_top=False)
        nweights = nm.get_weights()

        for i in range(len(nweights)):
            minshape = np.minimum(nweights[i].shape, oweights[i].shape)
            newshape = tuple([slice(0, s) for s in minshape])
            nweights[i][newshape] = oweights[i][newshape]

        nm.set_weights(nweights)

        keras.models.save_model(nm, f'data/mobilenet_v2_base_{int(alpha*10)}.h5')


make_mobilenetv1_base_weights()
make_mobilenetv2_base_weights()
