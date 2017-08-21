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
import os
import sys
import time

import imageio
import PIL
import pygame
import pygame.camera


def predict_caption(network, image):
    predictions = network.classify(image)
    best_guess = predictions[0]
    (label_id, label_name, confidence) = best_guess
    caption = label_name + " ({0:.0f}%)".format(confidence * 100.0)
    return caption


class InputWebcam:

    def open(self):
        pygame.camera.init()
        cameras = pygame.camera.list_cameras()
        self.camera_size = (640, 480)
        self.cam = pygame.camera.Camera(cameras[0], self.camera_size)
        self.cam.start()
        # Wait for the camera to warm up. I don't know why this is necessary, but
        # all of the examples seem to do it, no matter which camera API they use.
        time.sleep(0.1)

    def poll(self):
        surface = self.cam.get_image()
        image_bits = pygame.image.tostring(surface, "RGB", False)
        return (image_bits, self.camera_size)

    def close(self):
        self.cam.stop()


class OutputScreen:

    def __init__(self, network):
        self._network = network
        pygame.init()
        self._font = pygame.font.SysFont("monospace", 24, bold=True)
        self._window_size = (640, 480)
        self._aspect = self._window_size[0] / self._window_size[1]
        self._window = pygame.display.set_mode(self._window_size)
        pygame.display.set_caption("Plaidvision")
        self._running = True

    def running(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
        return self._running

    def process(self, image_bits, image_size):
        image = PIL.Image.frombytes("RGB", image_size, image_bits)
        # Convert the image into a surface object we can blit to the pygame window.
        surface = pygame.image.fromstring(image_bits, image_size, "RGB")
        # Fit the image proportionally into the window.
        (window_width, window_height) = self._window_size
        (image_width, image_height) = image_size
        result_size = image_size
        # If the image is larger than the window, in both dimensions, scale it
        # down proportionally so that it entirely fills the window.
        if image_width > window_width and image_height > window_height:
            hscale = float(window_width) / float(image_width)
            vscale = float(window_height) / float(image_height)
            if hscale > vscale:
                result_size = (window_width, int(image_height * hscale))
            else:
                result_size = (int(image_width * vscale), window_height)
        # Center the image in the window, cropping if necessary.
        hoff = (window_width - result_size[0]) / 2
        voff = (window_height - result_size[1]) / 2
        surface = pygame.transform.scale(surface, result_size)
        self._window.blit(surface, (hoff, voff))
        # Print some text explaining what we think the image contains, using some
        # contrasting colors for a little drop-shadow effect.
        caption = predict_caption(self._network, image)
        self.blit_caption(caption, -1, (110, 110, 240))
        self.blit_caption(caption, 2, (0, 0, 100))
        self.blit_caption(caption, 1, (100, 100, 255))
        self.blit_caption(caption, 0, (240, 240, 110))
        # Reveal the composited window buffer.
        pygame.display.flip()

    def blit_caption(self, caption, offset, color):
        label = self._font.render(caption, 1, color)
        label_pos = (32 + offset, self._window_size[1] - 48 + offset)
        self._window.blit(label, label_pos)


def has_plaid():
    try:
        import plaidml.keras
        return True
    except ImportError:
        return False


SUPPORTED_NETWORKS = ['inception_v3', 'mobilenet', 'resnet50', 'vgg16', 'vgg19', 'xception']


def main():
    parser = argparse.ArgumentParser()
    plaidargs = parser.add_mutually_exclusive_group()
    plaidargs.add_argument('--plaid', action='store_true')
    plaidargs.add_argument('--no-plaid', action='store_true')
    parser.add_argument('network', choices=SUPPORTED_NETWORKS)
    args = parser.parse_args()

    if args.plaid or (not args.no_plaid and has_plaid()):
        print("Using PlaidML backend.")
        import plaidml.keras
        plaidml.keras.install_backend()

    network = importlib.import_module('.'.join(['networks', args.network]))

    input = InputWebcam()
    output = OutputScreen(network)

    input.open()
    while output.running():
        (image_bits, image_size) = input.poll()
        if image_bits:
            output.process(image_bits, image_size)
    input.close()


if __name__ == "__main__":
    main()
