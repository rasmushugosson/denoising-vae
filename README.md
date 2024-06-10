# Denoising VAE

This project was written for our bachelor's degree project in computer science at KTH in Stockholm, Sweden.

## Getting started

*Details will be published later.*

### Set up remote Docker environment for GPU-accelerated training

This installation guide is based on [this article](https://www.pendragonai.com/setup-tensorflow-gpu-windows-docker-wsl2/) from Pendragon AI.

The only modification we made was to replace these commands:

```bash
docker pull tensorflow/tensorflow:2.14.0-gpu

docker run --name my_tensorflow -it --gpus all tensorflow/tensorflow:2.14.0-gpu bash
```

With these instead:

```bash
docker pull tensorflow/tensorflow:2.16.1-gpu

docker run --name my_tensorflow -it --gpus all tensorflow/tensorflow:2.16.1-gpu bash

```

To get the address of the container, run:

```bash
docker-compose logs image-ldm | grep -o 'http://127.0.0.1:8888/?token=[^ ]*' | tail -n1
```

## Licensing

Portions of this project are modifications of the [Keras Team's example of a Variational Autoencoder](https://github.com/keras-team/keras-io/blob/master/examples/generative/vae.py) by [fchollet](https://twitter.com/fchollet) and used according to terms described in the Apache 2.0 License. A copy of the Apache License is included in this repository in the LICENSE file.
