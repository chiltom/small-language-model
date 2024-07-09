# Notes

## `torch` Package Explanations

`torch.stack(tensors, dim=0, *) -> Tensor`: Concatenates a sequence of tensors along a new dimension.

`torch.multinomial(input, num_samples, replacement=False, *) -> LongTensor`: Returns a tensor where each row contains `num_samples` indices sampled from the multinomial probability distribution located in the corresponding row of tensor `input`.

- If `input` is a vector, `out` is a vector of size `num_samples`.
- If `input` is a matrix with _m_ rows, `out` is a matrix of shape (_m_ x `num_samples`).
- If `replacement` is `True`, samples are drawn without replacement. If not, they are drawn without replacement, which means that when a sample index is drawn for a row, it cannot be drawn again for that row.

`torch.zeros(*size, *) -> Tensor`: Returns a tensor filled with the scalar value _0_, with the shape defined by the variable argument `size`.

`torch.arange(start=0, end, step=1, *) -> Tensor`: Returns a 1-D tensor of size ((end-start)/step) with values from the interval [`start`, `end`) taken with common difference `step` beginning from `start`.

`torch.transpose(input, dim0, dim1) -> Tensor`: Returns a tensor that is a transposed version of `input`. The given dimensions `dim0` and `dim1` are swapped.

`torch.no_grad(orig_func=None)`: Context-manager that disables gradient calculation.

- Disabling gradient calculation is useful for inference, when you are sure that you will not call `Tensor.backward()`. It will reduce memory consumption for computations that would otherwise have `requires_grad=True`.

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

`nn.ModuleList(modules=None)`: Holds submodules in a list. `ModuleList` can be indexed like a regular Python list, but modules it contains are properly registered, and will be visible by all `Module` methods.

- `modules` (iterable, optional): an iterable of modules to add.

`nn.Embedding(num_embeddings, embedding_dim)`: A simple lookup table that stores embeddings of a fixed dictionary and size.

- `num_embeddings` (int): size of the dictionary of embeddings.
- `embedding_dim` (int): the size of each embedding vector.

`nn.LayerNorm(normalized_shape)`: Applies Layer Normalization over a mini-batch of inputs. The mean and standard-deviation are calculated over the last _D_ dimensions, where _D_ is the dimension of `normalized_shape`.

- `normalized_shape` (int or list or `torch.Size`): input shape from an expected input of size.

`nn.ReLU(inplace=False)`: Applies the rectified linear unit function element-wise.

- $ReLU(x) = (x)^+ = max(0,x)$

`nn.Dropout(p=0.5, inplace=False)`: During training, randomly zeroes some of the elements of the input tensor with probability `p`. The zeroed elements are chosen independently for each forward call and are sampled from a Bernoulli distribution.

- This has proben to be an effective technique for regularization and preventing the co-adaptation of neurons.

`nn.init.normal_(tensor, mean=0.0, std=1.0) -> Tensor`: Fill the input Tensor with values drawn from the normal distribution.

- `tensor` (Tensor): An n-dimensional `torch.Tensor`.
- `mean` (float): The mean of the normal distribution.
- `std` (float): The standard deviation of the normal distribution.

`nn.init.zeros_(tensor) -> Tensor`: Fill the input Tensor with the scalar value _0_.

## `torch.nn.functional` Package Explanations

`F.cross_entropy(input, target)`: Compute the cross entropy loss between input logits and target.

- `input` (Tensor): Predicted unnormalized logits.
- `target` (Tensor): Ground truth class indices or class probabilities.

`F.softmax(input, dim=None) -> Tensor`: Apply a softmax function. It is applied to all slices along `dim`, and will re-scale them so that the elements lie in the range [0, 1] and sum to 1.

- `input` (Tensor): input.
- `dim` (int): A dimension along which softmax will be computed.

## `torch.nn.Module` Class and Method Explanations

> [!NOTE]
> This class is the base class for all neural network modules. Your models should also subclass this class.
> Modules can also contain other Modules, allowing for nesting them in a tree structure. You can assign the submodules as regular attributes. Submodules assigned in this way will be registered, and will have their parameters converted too when you call `to()`, etc.

`to(device=None, dtype=None) -> self`: Loads the model onto the CPU or onto the GPU using CUDA and/or casts the parameters and buffers to the specified `dtype`.

`train(mode=True) -> self`: Set the module in **training** mode.

- `mode` (bool): Whether to set training mode (`True`) or evaluation mode (`False`).

`eval() -> self`: Set the module in **evaluation** mode.

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

**logits**: Logits are the **unnormalized** outputs of a neural network. A softmax (normalization) function is used to squash the outputs of a neural network (logits) so that they are all between 0 and 1 and sum to 1.

Multi-head attention allows for the parallel training of many different blocks, or decoders in this case, to efficiently train the model while capturing a good variety of predictions. The different heads are trained with scaled dot-product attention, their results are concatenated, a Linear transformation is applied to the converged results, and finally a dropout is applied to the transformation result to prevent overfitting.

Applying the softmax function makes significant values stand out more, giving them an "attention score" that is taken into consideration when making generated predictions based on input. It allows the model to learn more from and train better on important tokens.

Using `nn.ModuleList` in some locations and `nn.Sequential` in others is done with a specific purpose.
`nn.ModuleList` is a container that houses multiple `nn.Module`s that _do not_ depend on each other. By registering CUDA Modules in the ModuleList, it can be used to run all of these modules in parallel. This allows for more efficient training and computation, but uses more GPU memory as the input size and number of blocks increases.
`nn.Sequential` is a container that houses multiple `nn.Module`s that _do_ depend on one another. The sequence of Modules placed in the Sequential container matters, and all computations and training runs synchronously starting with the first Module. Each sub-Module (excluding the first) within the Sequential container depends on the input of the last Module, uses that input in its computations and training, and submits the output to the next Module.

### Broadcasting Semantics

In short, if a PyTorch operation supports broadcast, then its Tensor arguments can be automatically expanded to be of equal sizes (without making copies of the data).

Two tensors are "broadcastable" if the following rules hold:

- Each tensor has at least one dimension.
- When iterating over the dimension sizes, starting at the trailing dimension, the dimension sizes must either be equal, one of them is 1, or one of them does not exist.

If two tensors `x`, `y` are "broadcastable", the resulting tensor size is calculated as follows:

- If the number of dimensions of `x` and `y` are not equal, prepend 1 to the dimensions of the tensor with fewer dimensions to make them equal length.
- Then, for each dimension size, the resulting dimension size is the max of the sizes of `x` and `y` along that dimension.

## Further Reading

**Efficiency Testing**: Timing how long operations take using the `time` package. Start timing how long it takes to train the model, how long it takes to calculate loss, how long different sets of hyper-parameters take, etc.

**LLM and Transformer/GPT History**

- A Survey of Large Language Models

**Quantization**: Reducing memory usage by adjusting parameters.

- QLoRA: Efficient Finetuning of Quantized LLMs

**Gradient Accumulation**: Accumulates a gradient over _x_ iterations and averages it, updating parameters every _x_ iterations instead of every iteration.

**Hugging Face**: Different models, datasets, and documentation for reference and use online. Good for further research, development, and concentration.

## References

Brownlee, J. (2021) Gentle introduction to the adam optimization algorithm for deep learning, MachineLearningMastery.com. Available at: https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/ (Accessed: 07 July 2024).

Papers with code - AdamW explained. Available at: https://paperswithcode.com/method/adamw (Accessed: 07 July 2024).

PyTorch documentation: https://pytorch.org/docs/stable/index.html.
