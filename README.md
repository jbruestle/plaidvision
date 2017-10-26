# plaidvision
Vision network demos

To install (this will install [PlaidML](https://github.com/vertexai/plaidml):

```
pip install -r requirements.txt
```

Run the webcam classifier demo like this:

`python plaidvision.py [--plaid|--no-plaid] [MODEL]`

You can either use the default model or specify one of the Keras application
models directly - model names are "inception_v3", "resnet50", "vgg16",
"vgg19", "xception", and "mobilenet".

By default, PlaidML is used as the backend. Specifying `--no-plaid` will utilize tensorflow,
if installed.

