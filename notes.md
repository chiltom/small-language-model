# Notes

## `torch` Package Explanations
`torch.stack(tensors, dim=0, *) -> Tensor`: Concatenates a sequence of tensors along a new dimension.

`torch.multinomial(input, num_samples, replacement=False, *) -> LongTensor`: Returns a tensor where each row contains `num_samples` indices sampled from the multinomial probability distribution located in the corresponding row of tensor `input`.
- If `input` is a vector, `out` is a vector of size `num_samples`.
- If `input` is a matrix with *m* rows, `out` is a matrix of shape (*m* x `num_samples`).
- If `replacement` is `True`, samples are drawn without replacement. If not, they are drawn without replacement, which means that when a sample index is drawn for a row, it cannot be drawn again for that row.

`torch.zeros(*size, *) -> Tensor`: Returns a tensor filled with the scalar value *0*, with the shape defined by the variable argument `size`.

## `torch.Tensor` Class and Method Explanations
A `torch.Tensor` is a multi-dimensional matrix containing elements of a single data type.

### Methods/Attributes
`Tensor.shape`: Returns the size of the `self` tensor.

`Tensor.view(*shape) -> Tensor`: Returns a new tensor with the same data as the `self` tensor but of a different `shape`.

## `torch.nn` Package Explanations
`nn.Linear(in_features, out_features, bias=True)`: Applies a linear transformation to the incoming data: $y = xA^T + b$.
- `in_features` (int): size of each input sample.
- `out_features` (int): size of each output sample.
- `bias` (bool): if set to `False`, the layer will not learn an additive bias.

`nn.Sequential(*args: Module)`: A sequential container. Modules will be added to it in the order they are passed in the constructor.
- The `forward()` method of `Sequential` accepts any input and forwards it to the first module it contains. It then "chains" outputs to inputs sequentially for each subsequent module, finally returning the output of the last module.
- The value a `Sequential` provides over manually calling a sequence of modules is that it allows treating the whole container as a single module, such that performing a transformation on the `Sequential` applies it to each of the modules it stores (which are a registered submodule of the `Sequental`).

`nn.Embedding(num_embeddings, embedding_dim)`: A simple lookup table that stores embeddings of a fixed dictionary and size.
- `num_embeddings` (int): size of the dictionary of embeddings.
- `embedding_dim` (int): the size of each embedding vector.

`nn.LayerNorm(normalized_shape)`: Applies Layer Normalization over a mini-batch of inputs. The mean and standard-deviation are calculated over the last *D* dimensions, where *D* is the dimension of `normalized_shape`.
- `normalized_shape` (int or list or `torch.Size`): input shape from an expected input of size.

## `torch.nn.functional` Package Explanations
`F.cross_entropy`: Compute the cross entropy loss between input logits and target.

`F.softmax`: Apply a softmax function.

## `torch.optim.AdamW` Class and Method Explanations
**Adam** (Adaptive Moment Estimation) is an adaptive learning rate algorithm designed specifically for training deep neural networks.
It has become the default optimization method in many machine learning tasks due to its fast convergence and robustness across various platforms.

To understand Adam, we need to start with the basics. Imagine you're training a neural network, and you want to update its model parameters (often denoted as θ) to minimize the loss function.
Standard gradient descent (GD) is the vanilla optimization algorithm. It contains a fixed learning rate, which causes problems when you might need to manually adjust them during training.
- Too high, and you overshoot.
- Too low, and convergence becomes slow.

**Adam** solves this issue by adapting the learning rate for each parameter individually. It combines the best properties of two other optimization methods: AdaGrad (Adaptive Gradient Algorithm) and RMSProp (Root Mean Square Propogation).

Adam works by computing individual adaptive learning rates for each parameter based on their gradient history. The key components are **momentum** and **RMS (Root Mean Square)**.
- Momentum: Adam maintains an exponentially moving average of past gradients (similar to RMSProp). This helps smooth out "noisy" gradients.
- RMS: It also keeps track of the square gradients.

**AdamW** is an enhancement to the traditional Adam optimization algorithm. It modifies the typical implementation of weight decay in Adam by decoupling it from the gradient update.
Instead of directly adding the weight decay term to the gradient, AdamW adjusts the weight decay term within the gradient update itself.

By decoupling weight decay, AdamW avoids the convergence issues associated with the original Adam algorithm. It also ensures that weight decay doesn't interfere with the adaptive learning rates computed by Adam.

### Methods
`torch.optim.AdamW.zero_grad(set_to_none=True)`: Resets the gradients of all optimized `torch.Tensor`s.

`torch.optim.AdamW.step(closure=None)`: Perform a single optimization step.

## Miscallaneous Explanations
`model.to(device)`: Loads the model onto the GPU using CUDA, which was defined at the top of the notebook.
**logits**: Logits are the **unnormalized** outputs of a neural network. A softmax (normalization) function is used to squash the outputs of a neural network (logits) so that they are all between 0 and 1 and sum to 1.

References: 
- Brownlee, J. (2021) Gentle introduction to the adam optimization algorithm for deep learning, MachineLearningMastery.com. Available at: https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/ (Accessed: 07 July 2024).
- Papers with code - AdamW explained. Available at: https://paperswithcode.com/method/adamw (Accessed: 07 July 2024).
- PyTorch documentation: https://pytorch.org/docs/stable/index.html.
