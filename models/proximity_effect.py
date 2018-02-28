#By: Faustin Carter (faustin.carter@gmail.com), 2017

import scipy.special as ss
import scipy.constants as sc
import scipy.optimize as so
import scipy.integrate as si
import numpy as np

#This helps with integrating for the trilayer model
from ..helper_functions import wrap_for_numba

def bilayer_model(params, xvals, data=None, axis='s', **kwargs):
    """Calculate a residual for SN bilayer tc data using the bilayer_kernel.
    
    Parameters
    ----------
    params : lmfit.Paramters object
        
    xvals : numpy.array
        This can either be an array of S film thickness (axis='s') or an array
        of N film thicknesses (axis='n'). Default is axis='s'. In either case, a
        single value for the other film type thickness must be included in
        params. Units are meters.
    
    data : numpy.array
        An array of critical temperature values, one for each value in xvals.
        Units are Kelvin. If no data is passed, then an array of tc values is
        returned, calculated at each of the xvals. Otherwise a residual is
        calculated.
        
    axis : str
        Either 's' for superconducting or 'n' for normal. This specifies what
        type of thicknesses xvals contains. If axis='s' then params must include
        'dN' and if axis='n' params must contain 'dS'.
        
    Keyword Arguments
    -----------------
    eps : float or numpy.array
        Uncertainty values on the data. Either a single value, or one for each
        value in data.
        
    quasi-trilayer : bool
        Whether to use a linear combination of thicknesses for the N film
        thickness. This allows an approximation of a trilayer model. This
        requires 'weight' and 'dN2' be in params. The normal thickness is then
        calculated as dN = dN + weight*dN2. This requires that axis='s'. Default
        is False.
        
    Returns
    -------
    retval : numpy.array
        If data is not None then a residual of the model calculated at each value
        in xvals is returned. Otherwise the model is returned."""

    assert axis in ['s', 'n'], "Must choose either superconducting or normal axis."
    eps = kwargs.pop('eps', None)
    
    quasi_trilayer = kwargs.pop('quasi-trilayer', False)
    
    if quasi_trilayer:
        assert all(p in params.keys() for p in ['dN2', 'weight']), "Extra params required. See docs."
        assert axis == 's', "Quasi-trilayer is only supported with axis type 's'."
        dSs = xvals
        dN1 = params['dN']
        dN2 = params['dN2']
        weight = params['weight']
        dN = dN1 + weight*dN2
    elif axis == 's':
        dSs = xvals
        dN = params['dN']
    elif axis == 'n':
        dNs = xvals
        dS = params['dS']
        
    tc0 = params['tc0']
    vFS = params['vFS']
    vFN = params['vFN']
    thetaD = params['thetaD']
    tInt = params['tInt']
    
    if (eps is None) and (data is not None):
        #For now ignore this, but allow for a real value to be included
        eps = 1
        
    if axis == 's':
        model = bilayer_kernel(dSs, dN, tc0, vFS, vFN, thetaD, tInt)
    elif axis == 'n':
        model = bilayer_kernel(dS, dNs, tc0, vFS, vFN, thetaD, tInt)

    if data is not None:
        residual = (model-data)/eps
        return residual
    else:
        return model

def trilayer_model(params, xvals, data=None, axis='s', **kwargs):
    """Calculate a residual for NSN trilayer tc data using the trilayer_kernel.
    
    Parameters
    ----------
    params : lmfit.Paramters object
        
    xvals : numpy.array
        This can either be an array of S film thickness (axis='s') or an array
        of N film thicknesses (axis='lhs' or 'rhs'). Default is axis='s'. In
        either case, a single value for each of the other two film types'
        thickness must be included in params. Units are meters.
    
    data : numpy.array
        An array of critical temperature values, one for each value in xvals.
        Units are Kelvin. If no data is passed, then an array of tc values is
        returned, calculated at each of the xvals. Otherwise a residual is
        calculated.
        
    axis : str
        Either 's' for superconducting or 'lhs' or 'rhs' for normal. This
        specifies what type of thicknesses xvals contains. If axis='s' then
        params must include 'dL' and 'dR' and if axis='lhs' or 'rhs' params must
        contain 'dS' and the other normal film param.  Default is 's'.
        
    Keyword Arguments
    -----------------
    eps : float or numpy.array
        Uncertainty values on the data. Either a single value, or one for each
        value in data.
        
    Returns
    -------
    retval : numpy.array
        If data is not None then a residual of the model calculated at each value
        in xvals is returned. Otherwise the model is returned."""

    assert axis in ['s', 'lhs', 'rhs'], "Must choose either superconducting or either normal axis."
    eps = kwargs.pop('eps', None)
    
    if axis == 's':
        dSs = xvals
        dL = params['dL']
        dR = params['dR']
    elif axis == 'lhs':
        dLs = xvals
        dS = params['dS']
        dR = params['dR']
    elif axis == 'rhs':
        dRs = xvals
        dS = params['dS']
        dL = params['dL']
        
    tc0 = params['tc0']
    vFS = params['vFS']
    vFL = params['vFL']
    vFR = params['vFR']
    thetaD = params['thetaD']
    tIntL = params['tIntL']
    tIntR = params['tIntR']
    tauL = params['tauL']
    tauR = params['tauR']
    
    
    if (eps is None) and (data is not None):
        #For now ignore this, but allow for a real value to be included
        eps = 1
        
    if axis == 's':
        model = trilayer_kernel(dSs, dL, dR, tc0, vFS, vFL, vFR, thetaD, tIntL, tIntR, tauL, tauR)
    elif axis == 'lhs':
        model = trilayer_kernel(dS, dLs, dR, tc0, vFS, vFL, vFR, thetaD, tIntL, tIntR, tauL, tauR)
    elif axis == 'rhs':
        model = trilayer_kernel(dS, dL, dRs, tc0, vFS, vFL, vFR, thetaD, tIntL, tIntR, tauL, tauR)

    if data is not None:
        residual = (model-data)/eps
        return residual
    else:
        return model

@np.vectorize
def bilayer_kernel(dS, dN, tc0, vFS, vFN, thetaD, tInt):
    """Calculate the critical temperature of a SN bilayer film.
    
    Parameters
    ----------
    dS : float
        Thickness of superconducting film in meters
    dN : float
        Thickness of normal film in meters
    tc0 : float
        Critical temperature of bulk superconducting film in Kelvin
    vFS : float
        Fermi velocity of superconducting film in meters/second
    vFN : float
        Fermi velocity of normal metal film in meters/second
    thetaD : float
        Debye temperature of superconducting film in Kelvin
    tInt : float
        Interface transparency, a number between 0 and 1
    
    Returns
    -------
    tc : float
        Critical temperature of the bilayer in Kelvin
        
    Note
    ----
    This method of calculating Tc is from:
    
    Superconducting properties of thin dirty superconductor-normal-metal bilayers
    Ya. V. Fominov and M. V. Feigel'man
    Phys. Rev. B, Vol. 63, 094518 (2001)
    Eqs. (25) and (30)
    
    The units have been converted from the Natural units used in the paper
    to SI units.
    """
    
    #Calculate Debye frequency(ish) it really should be h and not hbar...
    omegaD = sc.k*thetaD/sc.h
    
    #If there is no superconductor there is no transition!
    if dS == 0:
        assert dN != 0, "Can't have a zero thickness film!"
        tc = 0.0
    #Run the model
    else:
        #Function to find the zeros of
        def tr(tc):

            #Handle divide by zero error
            if dN == 0:
                retval = -np.log(tc0/tc)
            else:
                tauS = 8*sc.pi**2*dS/(tInt*vFS)
                tauN = 8*sc.pi**2*dN*vFN/(tInt*vFS**2)

                tauR = tauN/(tauS+tauN)

                retval = (tauR*(ss.digamma(0.5+sc.h/(2*sc.pi*sc.k*tc*tauS*tauR))-
                             ss.digamma(0.5)-
                             np.log(np.sqrt(1+1/(tauR*tauS*omegaD)**2)))
                        -np.log(tc0/tc))
            return retval

        #The brentq method will throw a value error if it can't find a zero crossing
        #in that case, we should assume Tc doesn't exists (i.e. it is fully suppressed)s
        try:
            tc = so.brentq(tr, 1e-12, 1.5*tc0)
        except ValueError:
            tc = 0.0
            
    return tc

def trilayer_integrand(args):
    """Integral required for calculating the trilayer Tc, runs from 0 to kB*thetaD.
    
    Parameters
    ----------
    args : numpy.array
        Numpy array of arguments in the following order:
         * E : Energy (integration variable)
         * tc : critical temperature in Kelvin
         * dS : Thickness of superconducting film
         * dL : Thickness of LHS normal film
         * dR : Thickness of RHS normal film
         * vFS : Fermi velocity of the superconductor
         * vFL : Fermi velocity of the LHS normal film
         * vFR : Fermi velocity of the RHS normal film
         * tIntL : Interface transparency of LHS interface
         * tIntR : Interface transparency of RHS interface
         * tauL : electron spin-flip time in LHS normal film (set to -1 to ignore)
         * tauR : electron spin-flip time in RHS normal film (set to -1 to ignore)
    
    Returns
    -------
    igrand : float
        The integrand calculated at energy E
        
    Note
    ----
    This equation is the RHS of Eq. (13) and uses Eqs. (14-16) from:
    
    Modeling Iridium-Based Trilayer and Bilayer Transition-Edge Sensors
    G. Wang, J. Beeman, C.L. Chang, et. al.
    IEEE Trans. Appl. Supercond., Vol. 27, No. 4, June 2017"""
    
    E, tc, dS, dL, dR, vFS, vFL, vFR, tIntL, tIntR, tauL, tauR = args
    
    betaL = 4*np.pi/(tIntL*vFS*sc.hbar)
    betaR = 4*np.pi/(tIntR*vFS*sc.hbar)
    
    #Handle divide by zero error
    if E == 0:
        igrand = 1/(2*sc.k*tc)
    else:
        #Some logic to simplify the equations if either RHS or LHS is zero thickness
        if (dL > 0) or (dR > 0):
            if dL != 0:
                if (tauL != -1) and (tauL > 0):
                    f1 = betaL-1/(1j*E-sc.hbar/tauL)*vFS/vFL*1/dL
                else:
                    f1 = betaL-1/(1j*E)*vFS/vFL*1/dL

            if dR != 0:
                if (tauR != -1) and (tauR > 0):
                    f2 = betaR-1/(1j*E-sc.hbar/tauR)*vFS/vFR*1/dR
                else:
                    f2 = betaR-1/(1j*E)*vFS/vFR*1/dR

            if dL == 0:
                fp = f2
            elif dR == 0:
                fp = f1
            else:
                fp = f1*f2/(f1+f2)

            fA = (1/(1-1j*E*dS*fp)).real
        else:
            fA = 1.0

        igrand = fA*np.tanh(E/(2*sc.k*tc))/E
    return igrand


#Use numba to create a C-language callback to pass to SciPy for
#much faster integration times
c_trilayer_integrand = wrap_for_numba(trilayer_integrand)

@np.vectorize
def trilayer_kernel(dS, dL, dR, tc0, vFS, vFL, vFR, thetaD, tIntL, tIntR, tauL=-1, tauR=-1):
    """Calculate the critical temperature of a NSN trilayer film.
    
    Parameters
    ----------
    dS : float
        Thickness of superconducting film in meters
    dL : float
        Thickness of LHS normal film in meters
    dR : float
        Thickness of RHS normal film in meters
    tc0 : float
        Critical temperature of bulk superconducting film in Kelvin
    vFS : float
        Fermi velocity of superconducting film
    vFL : float
        Fermi velocity of LHS normal metal film
    vFR : float
        Fermi velocity of RHS normal metal film
    thetaD : float
        Debye temperature of superconducting film in Kelvin
    tIntL : float
        Interface transparency of LHS, a number between 0 and 1
    tIntR : float
        Interface transparency of RHS, a number between 0 and 1
    tauL : float
        Electron spin-flip time of LHS normal film. If material is non-Paramagnetic
        (i.e. spin flip time is infinite, set to -1). Default is -1.
    tauR : float
        Electron spin-flip time of RHS normal film. If material is non-Paramagnetic
        (i.e. spin flip time is infinite, set to -1). Default is -1.
    
    Returns
    -------
    tc : float
        Critical temperature of the bilayer in Kelvin
        
    Note
    ----
    This method of calculating Tc implements Eq. (13) from:
    
    Modeling Iridium-Based Trilayer and Bilayer Transition-Edge Sensors
    G. Wang, J. Beeman, C.L. Chang, et. al.
    IEEE Trans. Appl. Supercond., Vol. 27, No. 4, June 2017"""
    
    #Function to find the zero crossings of
    def tr(tc):

        iargs = (tc, dS, dL, dR, vFS, vFL, vFR, tIntL, tIntR, tauL, tauR)
        integral, _ = si.quad(c_trilayer_integrand, 0, sc.k*thetaD, args=iargs)

        if integral == np.inf:
            retval = 0
        else:
            retval = np.log(tc/tc0)+integral
        
        return retval
    
    #The brentq method will throw a value error if it can't find a zero crossing
    #in that case, we should assume Tc doesn't exists (i.e. it is fully suppressed)s
    try:
        tc = so.brentq(tr, 1e-12, 1.5*tc0)
    except ValueError:
        tc = 0.0
            
    return tc


