#By: Faustin Carter (faustin.carter@gmail.com), 2017

import numpy as np
import lmfit as lf

def do_lmfit(xdata, ydata, fit_fn, params, **kwargs):
    """Run any fit from models on your data.

    Paramters
    ---------
    xdata : np.array
        The points at which to calculate the model.

    ydata : np.array
        The data to compare to the calculated model.

    fit_fn : callable
        Model function to pass to minimizer. Must have signature"""
    #pop model kwargs off the top
    model_kwargs = kwargs.pop('model_kwargs', {})

    #Override any of the default Parameter settings
    if kwargs is not None:
        for key, val in kwargs.items():
            #Allow for turning on and off parameter variation
            if '_vary' in key:
                key = key.split('_')[0]
                if key in params.keys():
                    assert ((val is True) or (val is False)), "Must pass bool for vary"
                    params[key].vary = val
            #Allow for overriding the range
            elif '_range' in key:
                key = key.split('_')[0]
                if key in params.keys():
                    assert len(val) == 2, "Must pass min and max for range! np.inf or -np.inf are OK."
                    params[key].min = val[0]
                    params[key].max = val[1]
            #Allow for overriding the default guesses
            elif key in params.keys():
                params[key].value = val

    minObj = lf.Minimizer(fit_fn, params, fcn_args=(xdata, ydata, **model_kwargs))
    fit_result = minObj.minimize()

    return fit_result
