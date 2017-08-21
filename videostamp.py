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

import argparse
import importlib
import json
import time

import numpy

import imageio
from PIL import Image, ImageDraw, ImageFont

SUPPORTED_NETWORKS = ['inception_v3', 'mobilenet', 'resnet50', 'vgg16', 'vgg19', 'xception']


def format_caption(predictions):
    best_guess = predictions[0]
    (label_id, label_name, confidence) = best_guess
    return label_name + " ({0:.0f}%)".format(confidence * 100.0)


def map_predictions(predictions):
    result = []
    for item in predictions:
        result.append(dict(label_id=item[0], label_name=item[1], confidence=float(item[2])))
    return result


def process_video(network, input, output, stop, results):
    for index in range(stop):
        frame = input.get_data(index)
        # The reader produces RGB8 images in the form of a numpy array.
        # Convert to PIL image so we can resample.
        image = Image.fromarray(numpy.uint8(frame))

        # Run the image through our classifier to generate predictions.
        start_time = time.time()
        predictions = network.classify(image)
        end_time = time.time()
        caption = format_caption(predictions)
        elapsed = end_time - start_time
        print('{}/{} time: {} prediction: {}'.format(index + 1, stop, elapsed, caption))
        record = dict(frame=index, elapsed=elapsed, predictions=map_predictions(predictions))
        results.append(record)

        if output:
            # Draw the caption over the source image and emit a new video frame.
            # We'll do a little drop shadow effect so the text has contrast whether
            # its background is dark or bright.
            draw = ImageDraw.Draw(image)
            font = ImageFont.load_default()
            offsets = [-1, 2, 1, 0]
            colors = [(110, 110, 240), (0, 0, 100), (100, 100, 255), (240, 240, 110)]
            for offset, color in zip(offsets, colors):
                point = (8 + offset, 8 + offset)
                (red, green, blue) = color
                draw.text(point, caption, font=font, fill=(red, green, blue, 255))
            # Convert the captioned PIL image back to a numpy array for imageio.
            frame = numpy.array(image)
            output.append_data(frame)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('--output')
    parser.add_argument('-v', '--verbose', type=int, nargs='?', const=3)
    parser.add_argument('--plaid', action='store_true')
    parser.add_argument('--network', choices=SUPPORTED_NETWORKS, default='mobilenet')
    parser.add_argument('--frames', type=int)
    parser.add_argument('--json')
    args = parser.parse_args()

    if args.plaid:
        print('Using PlaidML backend.')
        import plaidml.keras
        if args.verbose:
            plaidml._internal_set_vlog(args.verbose)
        plaidml.keras.install_backend()

    network = importlib.import_module('.'.join(['networks', args.network]))

    input = imageio.get_reader(args.input)
    print('imageio input metadata:')
    print(input.get_meta_data())

    output = None
    if args.output:
        fps = input.get_meta_data()['fps']
        output = imageio.get_writer(args.output, fps=fps)

    json_output = dict(results=[])
    stop = args.frames or input.get_length()
    try:
        process_video(network, input, output, stop, json_output['results'])
    except Exception as ex:
        json_output['exception'] = ex
        raise
    finally:
        if output:
            output.close()
        input.close()

        if args.json:
            with open(args.json, 'w') as file_:
                json.dump(json_output, file_)


if __name__ == '__main__':
    main()
