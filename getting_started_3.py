#!/usr/bin/env python

import numpy as np
import functools
import nifty5 as ift
import h5py
import sys


def find_parameters(sm_list, d, space, htlist, contactorlist, plotting=False):
    current_energy_value = np.inf
    for ii in range(2, 30):
        sky = ift.InverseGammaOperator(space, 0.5, 10 ** ii)
        lh = ift.PoissonianEnergy(d) @ sky
        pos = ift.full(lh.domain, 0)
        e = ift.EnergyAdapter(
            pos, ift.StandardHamiltonian(lh), want_metric=True)
        mini = ift.NewtonCG(
            ift.GradInfNormController(
                1e-7, convergence_level=3, iteration_limit=20))
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

        #if plotting:
        #    p = ift.Plot()
        #    p.add(
        #        skymodel, norm=LogNorm(), title=skymodel.log().integrate())
        #    p.add(pspec, label='Power analyze')
        #    fname = join(self.out, 'debug{}.png'.format(ii))
        #    p.output(name=fname, ny=1, xsize=16, ysize=8)
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
    with h5py.File(fname) as f:
        return np.array(f['bins'])


def cos_sq(theta):
    if theta >= 0 and theta < np.pi:
        return np.cos(theta)**2
    else:
        return 0.


if __name__ == '__main__':
    np.random.seed(23)

    # sky_domain = ift.RGSpace((300, 300), (2 / 300, 2 * np.pi / 200))
    sky_domain = ift.HPSpace(nside=2**3)
    energy_domain = ift.RGSpace((10, ))
    lambda_domain = ift.RGSpace((20, ), (0.1, ))
    time_domain = ift.RGSpace((50, ))
    position_space = ift.DomainTuple.make(
        (lambda_domain, time_domain, sky_domain,
         energy_domain))  # lambda_domain, time_domain))


    harmonic_space_sky = sky_domain.get_default_codomain()
    ht_sky = ift.HarmonicTransformOperator(harmonic_space_sky, sky_domain)
    power_space_sky = ift.PowerSpace(harmonic_space_sky)

    harmonic_space_energy = energy_domain.get_default_codomain()
    ht_energy = ift.HarmonicTransformOperator(harmonic_space_energy,
                                              energy_domain)
    power_space_energy = ift.PowerSpace(harmonic_space_energy)

    harmonic_space_time = time_domain.get_default_codomain()
    ht_time = ift.HarmonicTransformOperator(harmonic_space_time, time_domain)
    power_space_time = ift.PowerSpace(harmonic_space_time)

    harmonic_space_lambda = lambda_domain.get_default_codomain()
    ht_lambda = ift.HarmonicTransformOperator(harmonic_space_lambda,
                                              lambda_domain)
    power_space_lambda = ift.PowerSpace(harmonic_space_lambda)

    ht_list = [ht_lambda, ht_time, ht_sky, ht_energy]
    contr0 = ift.ContractionOperator(position_space, (0, 1, 3))
    contr1 = ift.ContractionOperator(position_space, (0, 1, 2))
    contr2 = ift.ContractionOperator(position_space, (0, 2, 3))
    contr3 = ift.ContractionOperator(position_space, (1, 2, 3))
    contractor_list = [contr3, contr2, contr0, contr1]

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

    dct_mu_time = {
        'target': power_space_time,
        'n_pix': 16,  # 64 spectral bins

        # Spectral smoothness (affects Gaussian process part)
        'a': 3,  # relatively high variance of spectral curbvature
        'k0': .4,  # quefrency mode below which cepstrum flattens

        # Power-law part of spectrum:
        'sm': -5,  # preferred power-law slope
        'sv': .5,  # low variance of power-law slope
        'im': -10,  # y-intercept mean, in-/decrease for more/less contrast
        'iv': .3,  # y-intercept variance
        'keys': ['tau_mu_time', 'phi_mu_time']
    }

    dct_nu_time = {
        'target': power_space_time,
        'n_pix': 16,  # 64 spectral bins

        # Spectral smoothness (affects Gaussian process part)
        'a': 3,  # relatively high variance of spectral curbvature
        'k0': .4,  # quefrency mode below which cepstrum flattens

        # Power-law part of spectrum:
        'sm': -5,  # preferred power-law slope
        'sv': .5,  # low variance of power-law slope
        'im': -10,  # y-intercept mean, in-/decrease for more/less contrast
        'iv': .3,  # y-intercept variance
        'keys': ['tau_nu_time', 'phi_nu_time']
    }

    dct_mu_energy = {
        'target': power_space_energy,
        'n_pix': 16,  # 64 spectral bins

        # Spectral smoothness (affects Gaussian process part)
        'a': 3,  # relatively high variance of spectral curbvature
        'k0': .4,  # quefrency mode below which cepstrum flattens

        # Power-law part of spectrum:
        'sm': -5,  # preferred power-law slope
        'sv': .5,  # low variance of power-law slope
        'im': -10,  # y-intercept mean, in-/decrease for more/less contrast
        'iv': .3,  # y-intercept variance
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
        'sm': -5,  # preferred power-law slope
        'sv': .5,  # low variance of power-law slope
        'im': -10,  # y-intercept mean, in-/decrease for more/less contrast
        'iv': .3,  # y-intercept variance
        'im': -10,  # y-intercept mean, in-/decrease for more/less contrast
        'iv': .3,  # y-intercept variance
        'keys': ['tau_nu_energy', 'phi_nu_energy']
    }
    dct_mu_lambda = {
        'target': power_space_lambda,
        'n_pix': 16,  # 64 spectral bins

        # Spectral smoothness (affects Gaussian process part)
        'a': 3,  # relatively high variance of spectral curbvature
        'k0': .4,  # quefrency mode below which cepstrum flattens

        # Power-law part of spectrum:
        'sm': -5,  # preferred power-law slope
        'sv': .5,  # low variance of power-law slope
        'im': -10,  # y-intercept mean, in-/decrease for more/less contrast
        'iv': .3,  # y-intercept variance
        'im': -10,  # y-intercept mean, in-/decrease for more/less contrast
        'iv': .3,  # y-intercept variance
        'keys': ['tau_mu_lambda', 'phi_mu_lambda']
    }

    dct_nu_lambda = {
        'target': power_space_lambda,
        'n_pix': 16,  # 64 spectral bins

        # Spectral smoothness (affects Gaussian process part)
        'a': 3,  # relatively high variance of spectral curbvature
        'k0': .4,  # quefrency mode below which cepstrum flattens

        # Power-law part of spectrum:
        'sm': -5,  # preferred power-law slope
        'sv': .5,  # low variance of power-law slope
        'im': -10,  # y-intercept mean, in-/decrease for more/less contrast
        'iv': .3,  # y-intercept variance
        'im': -10,  # y-intercept mean, in-/decrease for more/less contrast
        'iv': .3,  # y-intercept variance
        'keys': ['tau_nu_lambda', 'phi_nu_lambda']
    }

    A_nu_sky = ift.SLAmplitude(**dct_nu_sky)
    A_nu_energy = ift.SLAmplitude(**dct_nu_energy)
    A_nu_time = ift.SLAmplitude(**dct_nu_time)
    A_nu_lambda = ift.SLAmplitude(**dct_nu_lambda)
    rho_nu = MfCorrelatedFieldAntares(
        position_space, (A_nu_lambda, A_nu_time, A_nu_sky, A_nu_energy),
        'xi_nu')

    A_mu_sky = ift.SLAmplitude(**dct_mu_sky)
    A_mu_energy = ift.SLAmplitude(**dct_mu_energy)
    A_mu_time = ift.SLAmplitude(**dct_mu_time)
    A_mu_lambda = ift.SLAmplitude(**dct_mu_lambda)
    rho_mu = MfCorrelatedFieldAntares(
        position_space, (A_mu_lambda, A_mu_time, A_mu_sky, A_mu_energy),
        'xi_mu')

    # Apply a nonlinearity

    signal = ift.exp(rho_nu) + ift.exp(rho_mu)

    # Build the line-of-sight response and define signal response

    efficiency = ift.Field(ift.makeDomain(time_domain),
                           val=np.ones(time_domain.shape))
    # R = ift.makeOp(efficiency)
    R = ift.DiagonalOperator(diagonal=efficiency,
                             domain=position_space,
                             spaces=1)
    lamb = R(signal)
    # Specify noise
    data_space = position_space

    noise = .001
    N = ift.ScalingOperator(noise, data_space)

    # Generate mock signal and data

    mock_position = ift.from_random('normal', lamb.domain)
    data = lamb(mock_position)
    data = np.random.poisson(data.to_global_data().astype(np.float64))
    data = ift.Field.from_global_data(data_space, data)
    find_parameters([2,2,2,2], data, position_space, ht_list, contractor_list)

    sys.exit()
    # Minimization parameters
    ic_sampling = ift.GradientNormController(iteration_limit=100)
    ic_newton = ift.GradInfNormController(name='Newton',
                                          tol=1e-7,
                                          iteration_limit=5)
    minimizer = ift.NewtonCG(ic_newton)

    # Set up likelihood and information Hamiltonian
    likelihood = ift.PoissonianEnergy(data)(lamb)
    H = ift.StandardHamiltonian(likelihood, ic_sampling)

    initial_mean = ift.MultiField.full(H.domain, 0.)
    mean = initial_mean

    # number of samples used to estimate the KL
    N_samples = 6

    plot = ift.Plot()
    plot.add(contr0(data), title="sky data")
    plot.add(contr1(data), title="energy data")
    plot.add(contr2(data), title="time data")
    plot.add(contr3(data), title="lambda data")
    plot.output(ny=2, ysize=6, xsize=16, name='data' + '.png')

    plot = ift.Plot()
    plot.add(contr0(rho_nu.force(mock_position)),
             title="nu, marginalized over sky")
    plot.add(A_nu_sky.force(mock_position), title="nu_sky_power")
    plot.add(contr0(rho_mu.force(mock_position)),
             title="mu, marginalized over sky")
    plot.add(A_mu_sky.force(mock_position), title="mu_sky_power")
    plot.output(ny=2, ysize=6, xsize=16, name='truth.png')
    sys.exit()
    # Draw new samples to approximate the KL five times
    for i in range(5):
        # Draw new samples and minimize KL
        KL = ift.MetricGaussianKL(mean, H, N_samples)
        KL, convergence = minimizer(KL)
        mean = KL.position

        # Plot current reconstruction
        plot = ift.Plot()
        plot.add([A_nu_sky.force(KL.position),
                  A_nu_sky.force(mock_position)],
                 title="nu_sky_power")
        plot.add(contr0(rho_mu.force(KL.position)),
                 title="mu, marginalized over sky")
        plot.add([A_mu_sky.force(KL.position),
                  A_mu_sky.force(mock_position)],
                 title="mu_sky_power")
        plot.output(ny=2,
                    ysize=6,
                    xsize=16,
                    name='iteration_' + str(i) + '.png')
    # Draw posterior samples
    KL = ift.MetricGaussianKL(mean, H, N_samples)
    sc = ift.StatCalculator()
    for sample in KL.samples:
        sc.add(signal(sample + KL.position))
