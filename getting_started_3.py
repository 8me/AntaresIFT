#!/usr/bin/env python

import functools

import h5py
import healpy as hp
import km3pipe as kp
import numpy as np
import nifty5 as ift
from simplified_exp import sexp

from SImplifiedSLAmplitude import *

log = kp.logger.get_logger('IFT')
log.setLevel('INFO')


def plot_pspec(space, d, htlist, contractor_list, names):
    p = ift.Plot()
    for i, c in enumerate(contractor_list):
        ht = htlist[i]
        adder = ift.Adder(ift.Field.full(space, 1))
        pspec = ift.power_analyze(ht.adjoint(c(ift.log(adder(d)))))
        p.add(pspec, title=names[i])
    p.output(name='data_pspecs.png', ny=1, xsize=16, ysize=8)


def find_parameters(sm_list, d, space, htlist, contactorlist, plotting=False):
    current_energy_value = np.inf
    for ii in range(2, 30):
        sky = ift.InverseGammaOperator(space, 0.5, 10**ii)
        lh = ift.PoissonianEnergy(d) @ sky
        pos = ift.full(lh.domain, 0)
        e = ift.EnergyAdapter(pos,
                              ift.StandardHamiltonian(lh),
                              want_metric=True)
        mini = ift.NewtonCG(
            ift.GradInfNormController(1e-7,
                                      convergence_level=3,
                                      iteration_limit=20))
        e, _ = mini(e)
        skymodel = sky(e.position)

        im_list = []
        zm_list = []
        zmvar_list = []
        for jj, c in enumerate(contactorlist):
            ht = htlist[jj]
            pspec = ift.power_analyze(ht.adjoint(c(ift.log(skymodel))))
            t0 = np.log(pspec.domain[0].k_lengths[1])
            try:
                im = np.log(pspec.to_global_data()[1]) - t0 * sm_list[jj]
                im_list.append(im)
            except FloatingPointError:
                im_list.append(None)

            zm = skymodel.log().integrate()
            zm_list.append(zm)
            zmvar = (skymodel.log() * 0 + 2).integrate()
            zmvar_list.append(zmvar)

        if plotting:
            p = ift.Plot()
            p.add(skymodel, norm=LogNorm(), title=skymodel.log().integrate())
            p.add(pspec, label='Power analyze')
            fname = join(self.out, 'debug{}.png'.format(ii))
            p.output(name=fname, ny=1, xsize=16, ysize=8)
        if e.value > current_energy_value:
            break
        current_energy_value = e.value
    print('Found parameters:\nim = {}\nzm = {}\nzmvar = {}'.format(
        im_list, zm_list, zmvar_list))
    return im, zm, zmvar, skymodel


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
        return 1e-5


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

    data_space = ift.DomainTuple.make((sky_domain, energy_domain))
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

    ht_list = [ht_sky, ht_energy]
    contr0 = ift.ContractionOperator(data_space, (1, ))
    # contr1 = ift.ContractionOperator(position_space, (0, ))
    contr2 = ift.ContractionOperator(data_space, (0, ))
    # contr3 = ift.ContractionOperator(position_space, (1, 2, 3))
    contractor_list = [contr0, contr2]

    # Set up an amplitude operator for the field
    dct_nu_sky = {
        'target': power_space_sky,
        'n_pix': 16,  # 64 spectral bins

        # Spectral smoothness (affects Gaussian process part)
        'a': 2,  # relatively high variance of spectral curbvature
        'k0': .4,  # quefrency mode below which cepstrum flattens

        # Power-law part of spectrum:
        'sm': -3,  # preferred power-law slope
        'sv': .5,  # low variance of power-law slope
        'im': 10,  # y-intercept mean, in-/decrease for more/less contrast
        'iv': 2,  # y-intercept variance
        'keys': ['tau_nu_sky', 'phi_nu_sky']
    }

    dct_mu_sky = {
        'target': power_space_sky,
        'n_pix': 16,  # 64 spectral bins

        # Spectral smoothness (affects Gaussian process part)
        'a': 2,  # relatively high variance of spectral curbvature
        'k0': .4,  # quefrency mode below which cepstrum flattens

        # Power-law part of spectrum:
        'sm': -3,  # preferred power-law slope
        'sv': .5,  # low variance of power-law slope
        'im': 10,  # y-intercept mean, in-/decrease for more/less contrast
        'iv': 2,  # y-intercept variance
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
        'n_pix': 10,  # 64 spectral bins

        # Spectral smoothness (affects Gaussian process part)
        'a': 3,  # relatively high variance of spectral curbvature
        'k0': .4,  # quefrency mode below which cepstrum flattens

        # Power-law part of spectrum:
        'sm': -5,  # preferred power-law slope
        'sv': 2,  # low variance of power-law slope
        'im': 3,  # y-intercept mean, in-/decrease for more/less contrast
        'iv': 2,  # y-intercept variance
        'keys': ['tau_mu_energy', 'phi_mu_energy']
    }

    dct_nu_energy = {
        'target': power_space_energy,
        'n_pix': 10,  # 64 spectral bins

        # Spectral smoothness (affects Gaussian process part)
        'a': 3,  # relatively high variance of spectral curbvature
        'k0': .4,  # quefrency mode below which cepstrum flattens

        # Power-law part of spectrum:
        'sm': -5,  # preferred power-law slope
        'sv': 2,  # low variance of power-law slope
        'im': 3,  # y-intercept mean, in-/decrease for more/less contrast
        'iv': 2,  # y-intercept variance
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

    # A_nu_sky = ift.SLAmplitude(**dct_nu_sky)
    A_nu_sky = SimpSLAmplitude(**dct_nu_sky)
    A_nu_energy = SimpSLAmplitude(**dct_nu_energy)
    # A_nu_time = ift.SLAmplitude(**dct_nu_time)
    # A_nu_lambda = ift.SLAmplitude(**dct_nu_lambda)
    rho_nu = MfCorrelatedFieldAntares(position_space, (A_nu_sky, A_nu_energy),
                                      'xi_nu')

    # rho_nu = ift.CorrelatedField(position_space, A_nu_sky)
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
    sexp_op = sexp(position_space, 1)
    ift.extra.check_jacobian_consistency(
        sexp_op, ift.from_random('normal', sexp_op.domain))
    signal = sexp_op(rho_nu) + sexp_op(rho_mu)
    # signal = ift.exp(rho_nu)
    log.info('Signal ready')
    # Build the line-of-sight response and define signal response

    lamb = signal
    # Specify noise
    # data_space = position_space

    noise = .001
    N = ift.ScalingOperator(noise, data_space)

    # Generate mock signal and data
    mock_position = ift.from_random('normal', lamb.domain)
    # data = lamb(mock_position)
    # data = np.random.poisson(data.to_global_data().astype(np.float64))
    # data = ift.Field.from_global_data(data_space, data)
    # log.info('Mock data generated')

    file_data = get_data('./binned_zero_runs_healpix_low_res.h5')
    log.info(file_data.shape)
    data = ift.Field.from_global_data(data_space, file_data)
    log.info('File data loaded')

    plot_pspec(data_space, data, ht_list, contractor_list, ['sky', 'energy'])
    # data = contr0(data)

    # find_parameters([-2, -5],
    #                 data,
    #                 position_space,
    #                 ht_list,
    #                 contractor_list,
    #                 plotting=True)
    # Minimization parameters
    ic_sampling = ift.GradientNormController(iteration_limit=100)
    ic_newton = ift.GradInfNormController(name='Newton',
                                          tol=1e-15,
                                          iteration_limit=15)
    minimizer = ift.NewtonCG(ic_newton)
    log.info('Minimizer setup ready')

    # Set up likelihood and information Hamiltonian
    add_one = ift.Adder(ift.Field.full(lamb.target, 1.))
    likelihood = ift.PoissonianEnergy(data)(add_one(lamb))
    H = ift.StandardHamiltonian(likelihood, ic_sampling)

    # initial_mean = ift.MultiField.full(H.domain, 0.)
    mean = mock_position

    # number of samples used to estimate the KL
    N_samples = 8

    plot = ift.Plot()

    add_s = ift.Adder(ift.Field(ift.makeDomain(sky_domain), 1))
    add_e = ift.Adder(ift.Field(ift.makeDomain(energy_domain), 1))
    plot.add(contr0(data), title="sky data")
    plot.add(ift.log(add_s(contr0(data))), title="log 1 + sky data")
    plot.add(contr2(data), title="energy data")
    plot.add(ift.log(add_e(contr2(data))), title="log 1 + energy data")
    # plot.add(contr1(data), title="time data")
    # plot.add(contr3(data), title="lambda data")
    plot.output(ny=2, ysize=6, xsize=16, name='data' + '.png')

    try:
        plot = ift.Plot()
        plot.add(contr0(sexp_op(rho_nu.force(mock_position))),
                 title="nu, sky data")
        plot.add(A_nu_sky.force(mock_position), title="nu_sky_power")
        plot.add(contr0(sexp_op(rho_mu.force(mock_position))),
                 title="mu, sky data")
        plot.add(A_mu_sky.force(mock_position), title="mu_sky_power")
        plot.add(contr2(sexp_op(rho_nu.force(mock_position))),
                 title="nu, energy data")
        plot.add(A_nu_energy.force(mock_position), title="nu_energy_power")
        plot.add(contr2(sexp_op(rho_mu.force(mock_position))),
                 title="mu, energy data")
        plot.add(A_mu_energy.force(mock_position), title="mu_energy_power")
        plot.output(ny=2, ysize=6, xsize=16, name='truth.png')
    except:
        pass
    # Draw new samples to approximate the KL five times
    for i in range(25):
        KL = ift.MetricGaussianKL(mean, H, N_samples)

        # Plot current
        print(A_nu_sky.force(KL.position).val.min())
        print(A_nu_sky.force(KL.position).val.max())
        print(rho_nu.force(KL.position).val.min())
        print(rho_nu.force(KL.position).val.max())
        print(sexp_op(rho_nu).force(KL.position).val.min())
        print(sexp_op(rho_nu).force(KL.position).val.max())
        plot = ift.Plot()
        plot.add(A_nu_sky.force(KL.position), title="nu_sky_power")
        plot.add(contr0(sexp_op(rho_mu.force(KL.position))), title="mu, sky")
        plot.add(contr2(sexp_op(rho_mu.force(KL.position))),
                 title="mu, energy")
        plot.add(A_mu_sky.force(KL.position), title="mu_sky_power")
        # plot.add(contr0(ift.exp(rho_nu.force(KL.position))), title="nu, sky")
        plot.add(contr0(sexp_op(rho_nu.force(KL.position))),
                 title="nu, sky",
                 cmap=True)
        plot.add(contr2(sexp_op(rho_nu.force(KL.position))),
                 title="nu, energy")
        plot.output(ny=2,
                    ysize=6,
                    xsize=16,
                    name='iteration_' + str(i) + '.png')

        # Draw new samples and minimize KL
        KL, convergence = minimizer(KL)
        mean = KL.position

    # Draw posterior samples
    KL = ift.MetricGaussianKL(mean, H, N_samples)
    sc = ift.StatCalculator()
    for sample in KL.samples:
        sc.add(signal(sample + KL.position))
