# Algebraic-NCD
This package aims to provide algebraic descriptions of deep learning algorithms that capture all their metadata. It is based on the categorical approach of [*Neural Circuit Diagrams*](https://openreview.net/forum?id=RyZB4qXEgt), where independent data is described using Cartesian products and axes are described using hom-functors.

Algebraic descriptions of algorithms have many benefits. They allow for algorithms to be converted to and from diagrams, enabling architectures to be readily understood. Algebraic descriptions form a foundation for robust mathematical manipulations and analyses of algorithms. For instance, we can view [backpropagation](https://arxiv.org/abs/1711.10455) as a map between algebraic expressions. This lets us automatically generate a backpropagated expression. We can also use algebraic descriptions to estimate an algorithm's memory and computational costs, guiding improved design.

Algebraic descriptions provide the metadata for the compilation of algorithms into high-performance code. Furthermore, we can use the same compilation procedure for an algebraic expression and its derived backpropagated form. This can address a significant gap in the deep learning space. PyTorch, the most popular deep learning package, does not provide end-to-end compilation of models. Low-level implementations of algorithms can be [far faster](https://github.com/ggerganov/llama.cpp) but often lack PyTorch's backpropagation and training capabilities.

This package is a proof-of-concept. It allows terms to be composed into algorithms using composition, Cartesian products, and hom-functors. Currently, compilation into PyTorch for some algorithms is supported. The goal is to add various additional features over time. This includes a GUI to display and modify algorithms diagrammatically, optimized low-level compilation, functorial backpropagation, and much more.


## Current Features
- ``Shapes`` are terms which represent both data types (objects) and operations (morphisms). Shapes are categorical constructs, and therefore the package utilizes functors, monoidal products, etc. to construct expressions.

- ``Shapes`` have ``ConstructionRule``s enforce strictness, meaning only one form of certain isomorphic expressions are present. For example, Cartesian tuples of size 1 are reduced to their singular item.

- **Compound shapes** can be constructed from composition (@), Cartesian products (+), coproducts (|), and hom-functors/lifting (>>/<<).

- **Configurables** are shapes with a "pending" size. When composition is attempted, these pending sizes are set to the corresponding value, ensuring composition. Configurables allow us to work with operations with variadic inputs, and ensure the inputs and outputs of algorithms properly compose.

- **Broadcasting semantics** which copies, hom-functors, or configures shapes to ensure operations align.
    - This is found in `composition.py`. Broadcasting is defined using functors and natural transformations, ensuring consistency.

- **Compilation** into PyTorch is currently supported for some operations. This is a proof-of-concept for future compilation into low-level languages.
    - This is found in `marches.py` and `nn.py`. Currently, Linear, SoftMax and Einops operations are supported.

## Future Features
- **Diagramming and a GUI** will allow for algebraic expressions to be turned into diagrams and vice-versa, letting diagrams act as an effective means of communicating, implementing, understanding, and analyzing deep-learning models.

- **Computation / Memory Transfer Costs** of algorithms can be derived from the metadata of categorical expressions. We can implement an operation to derive these characteristics from algebraic expressions.

- **Low-Level Compilation**, we can use the extensive metadata provided by categorical expressions to optimize the arrangements of memory, fusion of operations, and other aspects of effective CUDA code in a manner not provided by the descriptions fed to PyTorch, or the interpreted code it runs.

- **Multi-Categories** which allow the expression of multiple operations, and the transfer of data between them. This will allow for [backpropagation](https://arxiv.org/abs/1711.10455) to be defined as a functor from one algebraic expression to another.

# Installation
This package requires [Python3.12](https://www.python.org/downloads/) as it extensively uses some of the recent typing features introduced (generics, pattern matching, and much more). To take full advantage of these features, enable Python 'Inlay Hints' in VSCode's Pylance user settings.

Once downloaded, change to this directory and run the below script;
```
python3.12 -m pip install --upgrade setuptools
python3.12 -m pip install .
```
