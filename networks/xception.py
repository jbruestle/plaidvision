#!/usr/bin/env python

# Copyright 2017 Vertex.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import keras.backend as K
import numpy as np
from keras.applications import xception

# Xception thinks it only works with tensorflow, but that's because it doesn't
# know PlaidML exists. We'll pretend to be tensorflow during initialization.
old_backend = K.backend
K.backend = lambda: "tensorflow"
model = xception.Xception(weights='imagenet')
K.backend = old_backend


def classify(img, top_n=5):
    target_size = (299, 299)
    if img.size != target_size:
        img = img.resize(target_size)
    import keras.preprocessing.image
    from keras.applications import xception
    data = keras.preprocessing.image.img_to_array(img)
    data = np.expand_dims(data, axis=0)
    data = xception.preprocess_input(data)
    predictions = model.predict(data)
    return xception.decode_predictions(predictions, top=top_n)[0]
