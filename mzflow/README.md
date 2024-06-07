# Matlab implementation of probabilistic modeling of tabular data with normalizing flows

A Matlab reimplementation of [PZFlow](https://github.com/jfcrenshaw/pzflow), following the same ideas and implementation, except using Matlab paradigms.
Many of the features of PZflow have not been implemented, but some other features have been added.

If your data consists of continuous variables in a table, mzflow can model the joint probability distribution of your data set, with additional conditional
(explanatory) variables (that can be discrete), with some limitations.  A normalizing flow is a type of neural network where the layers can be inverted, meaning
that outputs can be driven as 'inputs' to determine what the corresponding inputs would have been.  Typically this means that the number of inputs and outputs must be identical.
In the context of probability, the basic concept is that the CDF of the variables is modeled as a rational-quadratic spline with K knots on the interval (0,1), and the parameters for the spline
are determined from a simple dense relu neural network that depends on the other data dimensions and the conditions.  These layers are 'rolled' to change to the next data dimension, and
the layers are stacked as one would in any deep network.  The final flow acts as a joint distribution, where the forward direction models the inverse CDF and the inverse models the CDF.
This can be used to obtain random samples from the CDF, which can then be used to approximate various marginal and conditional results.  In normalizing flow literature, the name 'bijector'
is used for the stack of invertible layers, and 'latent distribution' is used for the random distribution that is used to determine the probabilities, which is classically a uniform distribution on (0,1),
but can be a different probability distribution.

If you use this package in your research, please cite the following source (since this is not my idea):

The paper
```bibtex
@misc{crenshaw2024,
      title={Probabilistic Forward Modeling of Galaxy Catalogs with Normalizing Flows}, 
      author={John Franklin Crenshaw and J. Bryce Kalmbach and Alexander Gagliano and Ziang Yan and Andrew J. Connolly and Alex I. Malz and Samuel J. Schmidt and The LSST Dark Energy Science Collaboration},
      year={2024},
      eprint={2405.04740},
      archivePrefix={arXiv},
      primaryClass={astro-ph.IM}
}
```

The package needs considerable work to be fully documented in Matlab style. The example file test_flow.m and the brief description below give some ideas on how to use this.

## Flow

The class `Flow` is all that is needed to get started.  The constructor takes six arguments: the names of the input, conditions, and weight columns, a predefined bijector
and latent distribution (for advanced use), and if the conditional columns should be scaled.  At a minimum, only one data column name needs to be supplied.

A default latent distribution can be attached using `flow.set_default_latent()`, which is the uniform distribution, and a default bijector can be attached with `flow.set_default_latent(data)`.
The default bijector consists of `n` pairs of neural spline layers and rolls with one data variable per pair. The splines have 32 knots and two hidden layers in the internal network.
The data limits are set to the ceiling and floor of each dimension's minimum and maximum data values.

The flow can then be trained using `flow.train(data)`, and once it is trained, `flow.sample()` can be used to obtain random samples while `flow.posterior()` can be used to obtain conditional
distributions along a data column.
