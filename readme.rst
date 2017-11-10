Models for fitting various physical models against data.
========================================================
Wraps lmfit, which is a wrapper for scipy.optimize.

Models that exist:
 * Superconducting transition (resistance vs. temperature)
 * Complex IQ transmission resonance (S21 vs. frequency)
 * NS and NSN proximity effect models for calulating Tc vs film thickness

Each model must contain at least one method that implements a model and returns
a residual. Ideally, each model will also implement a parameter guessing method
that parses the data, makes some reasonable guesses and returns an
lmfit.Paramters object. Signatures of functions must be as follows::

  model_function(params, xvals, yvals=None, **kwargs)
      #code goes here
      return residual

  params_function(*args, **kwargs)
      params = lmfit.Parameters()
      #code goes here to modify params
      return params

It is highly encouraged to have your model function also have an option to
return a model calculated at the xvals if no data is passed, and it is good
practice to weight the residual with some uncertainty so that chi-squared values
make sense.

It is also highly encouraged to include docstrings in both functions to describe
the options.

Running an actual fit is as simple as calling the do_lmfit function from
fit_tools.py and passing it a model function, a parameters object, the data, and
any extra keywords that the model supports.
