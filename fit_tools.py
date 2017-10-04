import numpy as np
import lmfit as lf

def do_lmfit(xdata, ydata, fit_fn, params, **kwargs):

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
            else:
                raise ValueError("Unknown keyword: "+key)

    minObj = lf.Minimizer(fit_fn, params, fcn_args=(ydata, xdata, eps))
    fit_result = minObj.minimize()

    return fit_result
