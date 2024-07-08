# Bigram Language Model

> An Educational Small LLM Creation

This is the Bigram Language Model, a small-scale Large-Language-Model that uses neural networks and machine learning to train and create responses.

## Developing

> [!WARNING]
> This program uses Cuda from the PyTorch library to utilize an NVIDIA GPU for parallel processing. If you are on another operating system or use another GPU, you may have to change the device to `mps` or research which service that PyTorch has implemented for your OS/GPU.

To get started with the source code and developing locally:

- Clone this repository to your local machine with the following command:

  ```shell
  git clone https://github.com/chiltom/small-language-model.git
  ```

- Run the following commands in order to install the required dependencies:

  ```shell
  python3 -m venv .venv

  # Mac
  source .venv/bin/activate

  # Windows
  .venv\Scripts\activate

  pip install -r requirements.txt
  ```

- Create an IPython kernel for your virtual environment to use with Jupyter with these commands:

  ```shell
  ipython kernel install --user --name=.venv
  python -m ipykernel install --user --name=.venv
  ```

- Finally, run the following command to start the Jupyter Notebook server:
  ```shell
  jupyter notebook
  ```

There are two ways you can run the code - inside of VSCode with the Jupyter extension, or in the browser.

- To use the notebook in your browser, go to http://localhost:8888/notebooks/bigram.ipynb

## Purpose

This application is an educational exploration of creating a neural network and using machine learning to create a large language model (LLM).

This is one of the first projects I will complete in my Machine Learning education, and this model has a long way to go. I intend to build more features as time goes on and my ML knowledge increases.

## Attribution

This model was created by following the instruction and teaching of Elliot Arledge, who gives great explanation behind a lot of the concepts involved in this model's creation.

A video of his instruction can be found [here](https://www.youtube.com/watch?v=UU1WVnMk4E8).
