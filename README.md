# Bigram Language Model

> An Educational Small LLM Creation

This is the Bigram language model, which uses a deep-learning neural network and transformer architecture for natural language processing (NLP). The model was primarily created using the PyTorch package.

## Developing

> [!WARNING]
> This program uses MPS from the PyTorch library to leverage a Mac Silicon GPU for parallel processing. If you are on a different operating system or use another GPU, you may need to change the device to `CUDA` or research the appropriate service that PyTorch provides for your OS/GPU.

To get started with the source code and developing/running locally:

- Clone this repository to your local machine with the following command:

  ```shell
  git clone https://github.com/chiltom/small-language-model.git
  ```

- Run the following commands to install the required dependencies:

  ```shell
  python3 -m venv .venv

  # Mac
  source .venv/bin/activate

  # Windows
  .venv\Scripts\activate

  pip install -r requirements.txt
  ```

- Create an IPython kernel for your virtual environment with these commands:

  ```shell
  ipython kernel install --user --name=.venv
  python -m ipykernel install --user --name=.venv
  ```

- To start the Jupyter Notebook server:

  ```shell
  jupyter notebook
  ```

- To run training splits:

  ```shell
  # Adjust hyper-parameters as necessary
  python3 training.py
  ```

- To run the chatbot:
  ```shell
  # Adjust hyper-parameters as necessary
  python3 chatbot.py
  ```

## Overview

This application is an educational exploration of using a neural network with deep learning to create a large language model (LLM).

A bigram language model is a type of statistical language model that predicts the probability of a word in a sequence based on the previous word. It considers pairs of consecutive words and predicts the likelihood of encountering a specific word given the preceding word in a text.

- "Word" in this context can mean an actual text word, a syllable, or a character. This specific model uses characters as a vocabulary for attention scoring and generation.

The model utilizes a transformer architecture, consisting of multiple layers of self-attention and feed-forward neural networks, to perform language modeling tasks. Over time I plan to train the model more and more to perform natural language processing (NLP) to explore which datasets and hyper-parameters make training the most efficient and worthwhile.

A high-level overview of the transformer architecture can be found [here](./docs/architecture.png)

Various notes on different modules, classes, methods, attributes, and explanations can be found [here](./docs/notes.md)

## Attribution

This model was initiated by following the instruction and teaching of Elliot Arledge, who provides excellent explanations of many of the concepts involved in this model's creation.

A video of his instruction can be found [here](https://www.youtube.com/watch?v=UU1WVnMk4E8).
