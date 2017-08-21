# plaidvision
Vision network demos

To install (this will install [PlaidML](https://github.com/vertexai/plaidml):

```
apt install ffmpeg python-h5py python-numpy python-pil python-pygame
pip install -r requirements.txt
```

Run the webcam classifier demo like this:

`python plaidvision.py [--plaid|--no-plaid] [NETWORK]`

You can either use the default network or specify one of the Keras application
networks directly - network names are "inception_v3", "resnet50", "vgg16",
"vgg19", "xception", and "mobilenet".

By default, PlaidML is used as the backend. Specifying `--no-plaid` will utilize tensorflow,
if installed.

