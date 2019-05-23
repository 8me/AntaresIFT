#!/usr/bin/env python

import numpy as np
import functools
import nifty5 as ift
import h5py
from tqdm import tqdm
import healpy as hp
import km3pipe as kp
import sys

log = kp.logger.get_logger('IFT')
log.setLevel('INFO')


def MfCorrelatedFieldAntares(target, amplitudes, name='xi'):
    tgt = ift.DomainTuple.make(target)

    hsp = ift.DomainTuple.make([tt.get_default_codomain() for tt in tgt])
    ht = ift.HarmonicTransformOperator(hsp, target=tgt[0], space=0)
    for i in range(1, len(tgt)):
        ht_add = ift.HarmonicTransformOperator(ht.target,
                                               target=tgt[i],
                                               space=i)
        ht = ht_add @ ht

    psp = [aa.target[0] for aa in amplitudes]
    pd = ift.PowerDistributor(hsp, psp[0], 0)
    for i in range(1, len(tgt)):
        pd_add = ift.PowerDistributor(pd.domain, psp[i], i)
        pd = pd @ pd_add

    spaces = np.arange(len(tgt))
    d = [
        ift.ContractionOperator(pd.domain, list(np.delete(spaces, i))).adjoint
        for i in range(len(tgt))
    ]

    a_l = [dd @ amplitudes[ii] for ii, dd in enumerate(d)]
    a = a_l[0]
    for i in range(1, len(tgt)):
        a = a * a_l[i]
    #a = a0 * a1
    A = pd @ a
    # For `vol` see comment in `CorrelatedField`
    import operator
    vol = functools.reduce(operator.mul, [sp.scalar_dvol**-0.5 for sp in hsp])
    return ht(vol * A * ift.ducktape(hsp, None, name))


def get_data(fname):
    with h5py.File(fname, 'r') as f:
        a = np.array(f['bins'])
        a = np.sum(a, axis=0)
    return a


def cos_sq(theta):
    if theta >= 0 and theta < np.pi / 2:
        return np.cos(theta)**2
    else:
        return 0.


def make_muon_distribution_array(nside):
    pixels = hp.nside2npix(nside)
    vfunc = np.vectorize(cos_sq)
    theta, _ = hp.pix2ang(nside, range(pixels))
    return vfunc(theta)


if __name__ == '__main__':
    # np.random.seed(23)
    log.info('Script started!')
    # sky_domain = ift.RGSpace((300, 300), (2 / 300, 2 * np.pi / 200))
    sky_domain = ift.HPSpace(nside=2**2)
    energy_domain = ift.RGSpace((10, ))
    # lambda_domain = ift.RGSpace((10, ), (0.2, ))
    # time_domain = ift.RGSpace((500, ))
    position_space = ift.DomainTuple.make(
        (sky_domain, energy_domain))  # lambda_domain, time_domain))

    harmonic_space_sky = sky_domain.get_default_codomain()
    ht_sky = ift.HarmonicTransformOperator(harmonic_space_sky, sky_domain)
    power_space_sky = ift.PowerSpace(harmonic_space_sky)

    harmonic_space_energy = energy_domain.get_default_codomain()
    ht_energy = ift.HarmonicTransformOperator(harmonic_space_energy,
                                              energy_domain)
    power_space_energy = ift.PowerSpace(harmonic_space_energy)

    # harmonic_space_time = time_domain.get_default_codomain()
    # ht_time = ift.HarmonicTransformOperator(harmonic_space_time, time_domain)
    # power_space_time = ift.PowerSpace(harmonic_space_time)
    log.info('Domain setup finished')
    # harmonic_space_lambda = lambda_domain.get_default_codomain()
    # ht_lambda = ift.HarmonicTransformOperator(harmonic_space_lambda,
    #                                           lambda_domain)
    # power_space_lambda = ift.PowerSpace(harmonic_space_lambda)
    # Set up an amplitude operator for the field
    dct_nu_sky = {
        'target': power_space_sky,
        'n_pix': 16,  # 64 spectral bins

        # Spectral smoothness (affects Gaussian process part)
        'a': 3,  # relatively high variance of spectral curbvature
        'k0': .4,  # quefrency mode below which cepstrum flattens

        # Power-law part of spectrum:
        'sm': -5,  # preferred power-law slope
        'sv': .5,  # low variance of power-law slope
        'im': -10,  # y-intercept mean, in-/decrease for more/less contrast
        'iv': .3,  # y-intercept variance
        'keys': ['tau_nu_sky', 'phi_nu_sky']
    }

    dct_mu_sky = {
        'target': power_space_sky,
        'n_pix': 16,  # 64 spectral bins

        # Spectral smoothness (affects Gaussian process part)
        'a': 3,  # relatively high variance of spectral curbvature
        'k0': .4,  # quefrency mode below which cepstrum flattens

        # Power-law part of spectrum:
        'sm': -5,  # preferred power-law slope
        'sv': .5,  # low variance of power-law slope
        'im': -10,  # y-intercept mean, in-/decrease for more/less contrast
        'iv': .3,  # y-intercept variance
        'keys': ['tau_mu_sky', 'phi_mu_sky']
    }

    # dct_mu_time = {
    #     'target': power_space_time,
    #     'n_pix': 16,  # 64 spectral bins
    #
    #     # Spectral smoothness (affects Gaussian process part)
    #     'a': 3,  # relatively high variance of spectral curbvature
    #     'k0': .4,  # quefrency mode below which cepstrum flattens
    #
    #     # Power-law part of spectrum:
    #     'sm': -5,  # preferred power-law slope
    #     'sv': .5,  # low variance of power-law slope
    #     'im': -10,  # y-intercept mean, in-/decrease for more/less contrast
    #     'iv': .3,  # y-intercept variance
    #     'keys': ['tau_mu_time', 'phi_mu_time']
    # }
    #
    # dct_nu_time = {
    #     'target': power_space_time,
    #     'n_pix': 16,  # 64 spectral bins
    #
    #     # Spectral smoothness (affects Gaussian process part)
    #     'a': 3,  # relatively high variance of spectral curbvature
    #     'k0': .4,  # quefrency mode below which cepstrum flattens
    #
    #     # Power-law part of spectrum:
    #     'sm': -5,  # preferred power-law slope
    #     'sv': .5,  # low variance of power-law slope
    #     'im': -10,  # y-intercept mean, in-/decrease for more/less contrast
    #     'iv': .3,  # y-intercept variance
    #     'keys': ['tau_nu_time', 'phi_nu_time']
    # }

    dct_mu_energy = {
        'target': power_space_energy,
        'n_pix': 16,  # 64 spectral bins

        # Spectral smoothness (affects Gaussian process part)
        'a': 3,  # relatively high variance of spectral curbvature
        'k0': .4,  # quefrency mode below which cepstrum flattens

        # Power-law part of spectrum:
        'sm': -2,  # preferred power-law slope
        'sv': .5,  # low variance of power-law slope
        'im': -10,  # y-intercept mean, in-/decrease for more/less contrast
        'iv': .3,  # y-intercept variance
        'keys': ['tau_mu_energy', 'phi_mu_energy']
    }

    dct_nu_energy = {
        'target': power_space_energy,
        'n_pix': 16,  # 64 spectral bins

        # Spectral smoothness (affects Gaussian process part)
        'a': 3,  # relatively high variance of spectral curbvature
        'k0': .4,  # quefrency mode below which cepstrum flattens

        # Power-law part of spectrum:
        'sm': -2,  # preferred power-law slope
        'sv': .5,  # low variance of power-law slope
        'im': -10,  # y-intercept mean, in-/decrease for more/less contrast
        'iv': .3,  # y-intercept variance
        'keys': ['tau_nu_energy', 'phi_nu_energy']
    }
    # dct_mu_lambda = {
    #     'target': power_space_lambda,
    #     'n_pix': 16,  # 64 spectral bins
    #
    #     # Spectral smoothness (affects Gaussian process part)
    #     'a': 3,  # relatively high variance of spectral curbvature
    #     'k0': .4,  # quefrency mode below which cepstrum flattens
    #
    #     # Power-law part of spectrum:
    #     'sm': -5,  # preferred power-law slope
    #     'sv': .5,  # low variance of power-law slope
    #     'im': -10,  # y-intercept mean, in-/decrease for more/less contrast
    #     'iv': .3,  # y-intercept variance
    #     'im': -10,  # y-intercept mean, in-/decrease for more/less contrast
    #     'iv': .3,  # y-intercept variance
    #     'keys': ['tau_mu_lambda', 'phi_mu_lambda']
    # }
    #
    # dct_nu_lambda = {
    #     'target': power_space_lambda,
    #     'n_pix': 16,  # 64 spectral bins
    #
    #     # Spectral smoothness (affects Gaussian process part)
    #     'a': 3,  # relatively high variance of spectral curbvature
    #     'k0': .4,  # quefrency mode below which cepstrum flattens
    #
    #     # Power-law part of spectrum:
    #     'sm': -5,  # preferred power-law slope
    #     'sv': .5,  # low variance of power-law slope
    #     'im': -10,  # y-intercept mean, in-/decrease for more/less contrast
    #     'iv': .3,  # y-intercept variance
    #     'im': -10,  # y-intercept mean, in-/decrease for more/less contrast
    #     'iv': .3,  # y-intercept variance
    #     'keys': ['tau_nu_lambda', 'phi_nu_lambda']
    # }

    A_nu_sky = ift.SLAmplitude(**dct_nu_sky)
    A_nu_energy = ift.SLAmplitude(**dct_nu_energy)
    # A_nu_time = ift.SLAmplitude(**dct_nu_time)
    # A_nu_lambda = ift.SLAmplitude(**dct_nu_lambda)
    rho_nu = MfCorrelatedFieldAntares(position_space, (A_nu_sky, A_nu_energy),
                                      'xi_nu')

    A_mu_sky = ift.SLAmplitude(**dct_mu_sky)
    A_mu_energy = ift.SLAmplitude(**dct_mu_energy)
    # A_mu_time = ift.SLAmplitude(**dct_mu_time)
    # A_mu_lambda = ift.SLAmplitude(**dct_mu_lambda)
    rho_mu = MfCorrelatedFieldAntares(position_space, (A_mu_sky, A_mu_energy),
                                      'xi_mu')
    log.info('Slope operator setup finished')
    # Apply a nonlinearity
    muon_distribution = ift.Field(ift.makeDomain(sky_domain),
                                  val=make_muon_distribution_array(
                                      sky_domain.nside))
    # R = ift.makeOp(efficiency)
    R = ift.DiagonalOperator(diagonal=muon_distribution,
                             domain=position_space,
                             spaces=0)

    signal = ift.exp(rho_nu) + R(ift.exp(rho_mu))

    log.info('Signal ready')
    # Build the line-of-sight response and define signal response

    lamb = signal
    # Specify noise
    data_space = position_space

    noise = .001
    N = ift.ScalingOperator(noise, data_space)

    # Generate mock signal and data
    # mock_position = ift.from_random('normal', lamb.domain)
    # data = lamb(mock_position)
    # data = np.random.poisson(data.to_global_data().astype(np.float64))
    # data = ift.Field.from_global_data(data_space, data)
    # log.info('Mock data generated')

    file_data = get_data('./test.h5')
    log.info(file_data.shape)
    data = ift.Field.from_global_data(data_space, file_data)
    log.info('File data loaded')
    # Minimization parameters
    ic_sampling = ift.GradientNormController(iteration_limit=100)
    ic_newton = ift.GradInfNormController(name='Newton',
                                          tol=1e-7,
                                          iteration_limit=15)
    minimizer = ift.NewtonCG(ic_newton)
    log.info('Minimizer setup ready')

    # Set up likelihood and information Hamiltonian
    likelihood = ift.PoissonianEnergy(data)(lamb)
    H = ift.StandardHamiltonian(likelihood, ic_sampling)

    initial_mean = ift.MultiField.full(H.domain, 0.)
    mean = initial_mean

    # number of samples used to estimate the KL
    N_samples = 4

    plot = ift.Plot()
    contr0 = ift.ContractionOperator(position_space, (1, ))
    # contr1 = ift.ContractionOperator(position_space, (0, ))
    contr2 = ift.ContractionOperator(position_space, (0, ))
    # contr3 = ift.ContractionOperator(position_space, (1, 2, 3))
    plot.add(contr0(data), title="sky data")
    plot.add(contr2(data), title="energy data")
    # plot.add(contr1(data), title="time data")
    # plot.add(contr3(data), title="lambda data")
    plot.output(ny=1, ysize=6, xsize=16, name='data' + '.png')

    # plot = ift.Plot()
    # plot.add(contr0(rho_nu.force(mock_position)),
    #          title="nu, marginalized over sky")
    # plot.add(A_nu_sky.force(mock_position), title="nu_sky_power")
    # plot.add(contr0(rho_mu.force(mock_position)),
    #          title="mu, marginalized over sky")
    # plot.add(A_mu_sky.force(mock_position), title="mu_sky_power")
    # plot.output(ny=2, ysize=6, xsize=16, name='truth.png')
    # Draw new samples to approximate the KL five times
    for i in range(5):
        # Draw new samples and minimize KL
        KL = ift.MetricGaussianKL(mean, H, N_samples)
        KL, convergence = minimizer(KL)
        mean = KL.position

        # Plot current reconstruction
        plot = ift.Plot()
        plot.add(A_nu_sky.force(KL.position), title="nu_sky_power")
        plot.add(contr0(rho_mu.force(KL.position)),
                 title="mu, marginalized over sky")
        plot.add(contr2(rho_mu.force(KL.position)),
                 title="mu, marginalized over sky")
        plot.add(A_mu_sky.force(KL.position), title="mu_sky_power")
        plot.add(contr0(rho_nu.force(KL.position)),
                 title="nu, marginalized over sky")
        plot.add(contr2(rho_nu.force(KL.position)),
                 title="nu, marginalized over sky")
        plot.output(ny=2,
                    ysize=6,
                    xsize=16,
                    name='iteration_' + str(i) + '.png')
    # Draw posterior samples
    KL = ift.MetricGaussianKL(mean, H, N_samples)
    sc = ift.StatCalculator()
    for sample in KL.samples:
        sc.add(signal(sample + KL.position))
