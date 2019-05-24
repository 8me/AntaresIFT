import nifty5 as ift
from simplified_exp import sexp
import numpy as np


def SimpSLAmplitude(*, target, n_pix, a, k0, sm, sv, im, iv, keys=['tau', 'phi']):

    s = sexp(ift.makeDomain(target), 2)
    log_amp = LinearSLAmplitude(target=target, n_pix=n_pix, a=a, k0=k0, sm=sm,
                             sv=sv, im=im, iv=iv, keys=keys)
    return s @ log_amp


def LinearSLAmplitude(*, target, n_pix, a, k0, sm, sv, im, iv, keys=['tau', 'phi']):
    '''Operator for parametrizing smooth amplitudes (square roots of power
    spectra).

    The general guideline for setting up generative models in IFT is to
    transform the problem into the eigenbase of the prior and formulate the
    generative model in this base. This is done here for the case of an
    amplitude which is smooth and has a linear component (both on
    double-logarithmic scale).

    This function assembles an :class:`Operator` which maps two a-priori white
    Gaussian random fields to a smooth amplitude which is composed out of
    a linear and a smooth component.

    On double-logarithmic scale, i.e. both x and y-axis on logarithmic scale,
    the output of the generated operator is:

        AmplitudeOperator = 0.5*(smooth_component + linear_component)

    This is then exponentiated and exponentially binned (in this order).

    The prior on the linear component is parametrized by four real numbers,
    being expected value and prior variance on the slope and the y-intercept
    of the linear function.

    The prior on the smooth component is parametrized by two real numbers: the
    strength and the cutoff of the smoothness prior
    (see :class:`CepstrumOperator`).

    Parameters
    ----------
    n_pix : int
        Number of pixels of the space in which the .
    target : PowerSpace
        Target of the Operator.
    a : float
        Strength of smoothness prior (see :class:`CepstrumOperator`).
    k0 : float
        Cutoff of smothness prior in quefrency space (see
        :class:`CepstrumOperator`).
    sm : float
        Expected exponent of power law.
    sv : float
        Prior standard deviation of exponent of power law.
    im : float
        Expected y-intercept of power law. This is the value at t_0 of the
        LogRGSpace (see :class:`ExpTransform`).
    iv : float
        Prior standard deviation of y-intercept of power law.

    Returns
    -------
    Operator
        Operator which is defined on the space of white excitations fields and
        which returns on its target a power spectrum which consists out of a
        smooth and a linear part.
    '''
    if not (isinstance(n_pix, int) and isinstance(target, ift.PowerSpace)):
        raise TypeError

    a, k0 = float(a), float(k0)
    sm, sv, im, iv = float(sm), float(sv), float(im), float(iv)
    if sv <= 0 or iv <= 0:
        raise ValueError

    et = ift.ExpTransform(target, n_pix)
    dom = et.domain[0]

    # Smooth component
    dct = {'a': a, 'k0': k0}
    smooth = ift.CepstrumOperator(dom, **dct).ducktape(keys[0])

    # Linear component
    sl = ift.SlopeOperator(dom)
    mean = np.array([sm, im + sm*dom.t_0[0]])
    sig = np.array([sv, iv])
    mean = ift.Field.from_global_data(sl.domain, mean)
    sig = ift.Field.from_global_data(sl.domain, sig)
    linear = sl @ ift.Adder(mean) @ ift.makeOp(sig).ducktape(keys[1])

    # Combine linear and smooth component
    loglog_ampl = 0.5*(smooth + linear)

    # Go from loglog-space to linear-linear-space
    return et @ loglog_ampl