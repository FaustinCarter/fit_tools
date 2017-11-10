#By: Faustin Carter (faustin.carter@gmail.com), 2017

import numba
import ctypes as ct
import warnings

#As of scipy 0.19, can use the built-in LowLevelCallable to make callbacks!
try:
    from scipy import LowLevelCallable
    LLC_EXISTS = True
except:
    LLC_EXISTS = False

def wrap_for_numba(func):
    """Uses numba to create a C-callback out of a function.
    Also includes a hack to fix a bug in the way SciPy parses input args in quad.
    This will probably break in future versions of SciPy.

    Parameters
    ----------
    func : python function
        Signature float(float[:])

    Returns
    -------
    new_cfunc : numba.cfunc object
        Signature float(int, pointer-to-array).

    Note
    ----
    The ``__name__`` and ``argtypes`` attributes of new_cfunc are modified here.
    """
    
    #First need to jit the function so cfunc can handle it.
    jitted_func = numba.jit("float64(float64[:])", nopython=True)(func)


    def c_func(n, a):
        """Simple wrapper to convert (int, pointer-to-array) args to (list) args.

        Parameters
        ----------
        n : C-language integer

        a : C-language pointer to array of type double.

        Returns
        -------
        func : python function
            Function signature is float(float[:])"""

        #unpack the pointer into an array
        args = numba.carray(a,n)

        return jitted_func(args)

    #Function signature required by numba
    #arguments are integer denoting length of array and pointer to array
    c_sig = numba.types.double(numba.types.intc, numba.types.CPointer(numba.types.double))

    #Use numba to create a c-callback
    new_cfunc = numba.cfunc(c_sig)(c_func)

    if LLC_EXISTS == True:
        #convert this into something that scipy.integrate.quad can work with
        return LowLevelCallable(new_cfunc.ctypes)
    else:
        warnings.warn("Falling back on legacy scipy behavior. Should upgrade to verion 0.19 or greater.", DeprecationWarning)
        #This is a hack to address a bug in scipy.integrate.quad for scipy versions < 0.19
        new_cfunc.ctypes.argtypes = (ct.c_int, ct.c_double)

        return new_cfunc.ctypes