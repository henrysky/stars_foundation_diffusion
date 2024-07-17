Abstract
===========

Transformers are often the go-to architecture to build foundation models that ingest a large amount of training data. 
But these models do not estimate the probability density distribution when trained on regression problems, yet obtaining full 
probabilistic outputs is crucial to many fields of science, where the probability distribution of the answer can be non-Gaussian 
and multimodal. In this work, we demonstrate that training a probabilistic model using a denoising diffusion head on top of 
the Transformer provides reasonable probability density estimation even for high-dimensional inputs. The combined 
Transformer+Denoising Diffusion model allows conditioning the output probability density on arbitrary combinations of inputs 
and it is thus a highly flexible density function emulator of all possible input/output combinations. We illustrate our
 Transformer+Denoising Diffusion model by training it on a large dataset of astronomical observations and measured labels of 
 stars within our Galaxy and we apply it to a variety of inference tasks to show that the model can infer labels accurately 
 with reasonable distributions.

Getting Started
================

This repository is to make sure all figures and results are reproducible by anyone easily for this paper🤗.

If Github has issue (or too slow) to load the Jupyter Notebooks, you can go
http://nbviewer.jupyter.org/github/henrysky/stars_foundation_diffusion/tree/main/

Dependencies
----------------

Python dependencies are listed in `requirements.txt`_.

.. _requirements.txt: requirements.txt

..

    ⚠️ You have to set ``magicnumber = nan`` in ``astroNN`` `configuration file`_ for the data reduction code to work properly.

..

    ⚠️ Using ``mps`` backend of ``PyTorch`` on Apple device is known to yield incorrect results. Please use ``cuda`` or ``cpu`` as backend.


.. _configuration file: https://astronn.readthedocs.io/en/latest/quick_start.html#configuration-file

Datasets
---------------

Datasets are available on `Zenodo`_ and should be placed in the folder named ``data_files`` under the root directory of this repository.

.. _Zenodo: https://zenodo.org/records/12738256

If you are planning to use the Docker image, the data files are already downloaded and placed in the correct folder in the container.

Docker Image
----------------

If you have `Docker`_ installed, you can use the `Dockerfile`_ to build a Docker image upon Pytorch container from `NVIDIA NGC Catalog`_ with all dependencies installed and data files downloaded.

.. _NVIDIA NGC Catalog: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
.. _Dockerfile: Dockerfile
.. _Docker: https://www.docker.com/

To build the Docker image called ``stars_foundation_diffusion``, run the following command in the root directory of this repository:

.. code-block:: bash

    docker build -t stars_foundation_diffusion .

To run the Docker container with all GPU available to the container named ``testing123``, run the following command:

.. code-block:: bash
    
    docker run --gpus all --name testing123 -it -e SHELL=/bin/bash --entrypoint bash stars_foundation_diffusion

Then you can attach to the container by running:

.. code-block:: bash

    docker exec -it testing123 bash

Now you can run all notebooks or training script inside the container

Jupyter Notebooks
--------------------------------------------------------

-   | `Dataset_Reduction.ipynb`_
    | The notebook contains code to generate the dataset used by this paper. 
    | Terabytes of (mostly gaia) data need to be downloaded in the process to construct the datasets.
-   | `DDPM.ipynb`_
    | The notebook contains code to train a simple denoising diffusion model
-   | `DDPM_Conditional.ipynb`_
    | The notebook contains code to train a simple conditional denoising diffusion model
-   | `Inference.ipynb`_
    | The notebook contains code to do inference
-   | `California_Housing.ipynb`_
    | The notebook contains code to train a model on California housing dataset for demonstration purpose.

.. _Dataset_Reduction.ipynb: Dataset_Reduction.ipynb
.. _Inference.ipynb: Inference.ipynb
.. _DDPM.ipynb: DDPM.ipynb
.. _DDPM_conditional.ipynb: DDPM_conditional.ipynb
.. _California_Housing.ipynb: California_Housing.ipynb

Python Script
--------------------------------------------------------

If you use this training script to train your own model, please notice that details of your system will be 
saved automatically in the model folder as ``training_system_info.txt`` for developers to debug should anything went wrong. 
Delete the file before you share your model with others if you concern about privacy. 

-   | `training.py`_
    | Python script to train the model.

.. _training.py: training.py

Models
--------------------------------------------------------

-   | ``model_torch`` is a trained `PyTorch`_ model
    | The model has ~3.7 millions parameters for the paper
-   | ``trained_california_model`` is a trained `PyTorch`_ model
    | The model has 20640 parameters trained on California housing dataset for demonstration purpose

.. _PyTorch: https://pytorch.org/

Graphics 
--------------------------------------------------------

All these graphics can be opened and edited by `draw.io`_.

-   | `encoder_ddpm.drawio`_
    | Source for Figure 1 in the paper, 


.. _encoder_ddpm.drawio: encoder_ddpm.drawio
.. _draw.io: https://draw.io/

Authors
===========

-  | **Henry Leung** - henrysky_
   | Department of Astronomy and Astrophysics, University of Toronto
   | Contact Henry: henrysky.leung [at] utoronto.ca

-  | **Jo Bovy** - jobovy_
   | Department of Astronomy and Astrophysics, University of Toronto
   | Contact Jo: bovy [at] astro.utoronto.ca

.. _henrysky: https://github.com/henrysky
.. _jobovy: https://github.com/jobovy

License
---------
This project is licensed under the MIT License - see the `LICENSE`_ file for details

.. _LICENSE: LICENSE
