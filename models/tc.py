import numpy as np
import lmfit as lf

__all__ = ['tc_model', 'tc_params']

def tc_model(params, data, temps, eps):
    """A fit function that fits a tanh-like superconducting transition

    Parameters
    ----------
    params : lmfit.Paramters object

    data : numpy.array
        resistance data

    temps: numpy.array
        temperature data

    eps : numpy.array
        uncertaintly on resistance data

    Returns
    -------
    residual or model : numpy.array
        If data is passed, then this calculates a residual against the model.
        Otherwise it returns the model calculated from the parameter values."""

    #normal resistance (Ohms)
    Rn = params['Rn'].value

    #parasitic series resistance (Ohms)
    Rp = params['Rp'].value

    #Critical temperature (K)
    Tc = params['Tc'].value

    #Transition width (K)
    dTc = params['dTc'].value

    #Check to see if the parameters object passed in contains keys for a transition with steps
    has_steps = all(key in params.keys() for key in ['R1f', 'Tc1', 'dTc1', 'R2f', 'Tc2', 'dTc2'])

    if has_steps:
        #Parameters describing the first step in the transition
        R1f = params['R1f'].value
        Tc1 = params['Tc1'].value
        dTc1 = params['dTc1'].value

        #Parameters describing the second step in the transition
        R2f = params['R2f'].value
        Tc2 = params['Tc2'].value
        dTc2 = params['dTc2'].value

        model = 0.5*(Rn-R1f*Rn-R2f*Rn)*(1+np.tanh(2*np.log(3)*(temps-Tc)/dTc))+Rp+0.5*(R1f*Rn)*(1+np.tanh(2*np.log(3)*(temps-Tc1)/dTc1))+0.5*(R2f*Rn)*(1+np.tanh(2*np.log(3)*(temps-Tc2)/dTc2))
    else:

        model = 0.5*Rn*(1+np.tanh(2*np.log(3)*(temps-Tc)/dTc))+Rp

    if data is not None:
        residual = (model-data)/eps
        return residual
    else:
        return model

def tc_params(rvals, tvals, **kwargs):
    """Build up a Parameters object to pass to the fitting function"""

    params = lf.Parameters()


    #Use the standard deviation of the last 10 data points to calculate the noise
    #Or some custom range
    eps_range = kwargs.pop('eps_range', (-10, -1))
    eps_range = slice(*eps_range)
    eps = np.std(rvals[eps_range])

    #Whether or not to fit a model with up to two steps
    has_steps = kwargs.pop('has_steps', False)

    Rpguess = np.mean(rvals[0:10])
    params.add('Rp', value=Rpguess, min=Rpguess-5*eps, max=Rpguess+5*eps, vary=False)

    Rnguess = (np.mean(rvals[-10:-1])-Rpguess)
    params.add('Rn', value=Rnguess, min=Rnguess-5*eps, max=Rnguess+5*eps, vary=False)

    Tcguess = tvals[np.argmax(rvals>(2*Rnguess+3*Rpguess)/3.0)]
    params.add('Tc', value=Tcguess, min=Tcguess*0.9, max=Tcguess*1.1, vary=True)

    #The following params aren't used in the model, but we want to have them around later
    #anyhow. NT stands for ninety-ten.

    #The point at which the resistance rises above 50% of Rn
    midTcNT = tvals[np.argmax(rvals>(Rnguess+2*Rpguess)/2.0)]
    params.add('midTcNT', value=midTcNT, vary=False)

    #Calculate the ninety-ten width of the curve
    tenPctLoc = tvals[np.argmax(rvals>(Rnguess*0.1+Rpguess))]
    ninetyPctLoc = tvals[np.argmax(rvals>(Rnguess*0.9+Rpguess))]
    dTcguess = ninetyPctLoc-tenPctLoc
    params.add('widthNT', value=dTcguess, vary=False)

    #Use that as the guess for the transition width
    params.add('dTc', value=dTcguess, min=0.0005, max=0.05)

    if has_steps:
        #Try and be semi-smart about guessing the step values if they are needed
        #These values are hand tuned for SPT3G TESs out of Ti/Au so may not work for everything
        params.add('R1f', value=0.15, min=0, max=0.5)
        params.add('R2f', value=0.1, min=0, max=0.5)
        params.add('Tc1', value=Tcguess-0.025, min=Tcguess-0.1, max = Tcguess)
        params.add('Tc2', value=Tcguess-0.03, min=0.38, max=Tcguess)
        params.add('dTc1', value=0.02, min=0.0005, max=0.2)
        params.add('dTc2', value=0.02, min=0.0005, max=0.2)

    return params
