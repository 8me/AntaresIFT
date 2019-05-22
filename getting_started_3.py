# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2013-2019 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

############################################################
# Non-linear tomography
#
# The signal is a sigmoid-normal distributed field.
# The data is the field integrated along lines of sight that are
# randomly (set mode=0) or radially (mode=1) distributed
#
# Demo takes a while to compute
#############################################################

import sys

import numpy as np
import functools
import operator
import nifty5 as ift

def MfCorrelatedFieldAntares(target, amplitudes, name='xi'):
    tgt = ift.DomainTuple.make(target)

    hsp = ift.DomainTuple.make([tt.get_default_codomain() for tt in tgt])
    ht1 = ift.HarmonicTransformOperator(hsp, target=tgt[0], space=0)
    ht2 = ift.HarmonicTransformOperator(ht1.target, target=tgt[1], space=1)
    ht = ht2 @ ht1

    psp = [aa.target[0] for aa in amplitudes]
    pd0 = ift.PowerDistributor(hsp, psp[0], 0)
    pd1 = ift.PowerDistributor(pd0.domain, psp[1], 1)
    pd = pd0 @ pd1

    dd0 = ift.ContractionOperator(pd.domain, 1).adjoint
    dd1 = ift.ContractionOperator(pd.domain, 0).adjoint
    d = [dd0, dd1]

    a0 = d[0] @ amplitudes[0]
    a1 = d[1] @ amplitudes[1]


    # a = [dd @ amplitudes[ii] for ii, dd in enumerate(d)]
    # a = functools.reduce(operator.mul, [a0, a1])
    a = a0 * a1
    A = pd @ a
    # For `vol` see comment in `CorrelatedField`
    vol = functools.reduce(operator.mul, [sp.scalar_dvol**-0.5 for sp in hsp])
    return ht(vol*A*ift.ducktape(hsp, None, name))


if __name__ == '__main__':
    np.random.seed(420)

    sky_domain = ift.RGSpace((300, 300), (2/300, 2*np.pi/200))
    energy_domain = ift.RGSpace(10)
    # lambda_domain = ift.RGSpace((20,), (0.5,))
    time_domain = ift.RGSpace((500,))
    position_space = ift.DomainTuple.make((sky_domain, energy_domain,)) # lambda_domain, time_domain))

    harmonic_space_sky = sky_domain.get_default_codomain()
    ht_sky = ift.HarmonicTransformOperator(harmonic_space_sky, sky_domain)
    power_space_sky = ift.PowerSpace(harmonic_space_sky)

    harmonic_space_energy = energy_domain.get_default_codomain()
    ht_energy = ift.HarmonicTransformOperator(harmonic_space_energy, energy_domain)
    power_space_energy = ift.PowerSpace(harmonic_space_energy)

    # Set up an amplitude operator for the field
    dct_sky = {
        'target': power_space_sky,
        'n_pix': 64,  # 64 spectral bins

        # Spectral smoothness (affects Gaussian process part)
        'a': 3,  # relatively high variance of spectral curbvature
        'k0': .4,  # quefrency mode below which cepstrum flattens

        # Power-law part of spectrum:
        'sm': -5,  # preferred power-law slope
        'sv': .5,  # low variance of power-law slope
        'im':  0,  # y-intercept mean, in-/decrease for more/less contrast
        'iv': .3,   # y-intercept variance
        'keys': ['tau_sky', 'phi_sky']
    }

    dct_energy = {
        'target': power_space_energy,
        'n_pix': 64,  # 64 spectral bins

        # Spectral smoothness (affects Gaussian process part)
        'a': 3,  # relatively high variance of spectral curbvature
        'k0': .4,  # quefrency mode below which cepstrum flattens

        # Power-law part of spectrum:
        'sm': -5,  # preferred power-law slope
        'sv': .5,  # low variance of power-law slope
        'im':  0,  # y-intercept mean, in-/decrease for more/less contrast
        'iv': .3,   # y-intercept variance
        'keys': ['tau_energy', 'phi_energy']
    }

    A_nu_sky = ift.SLAmplitude(**dct_sky)
    A_nu_energy = ift.SLAmplitude(**dct_energy)
    rho_nu = MfCorrelatedFieldAntares(position_space, (A_nu_sky, A_nu_energy))

    A_mu_sky = ift.SLAmplitude(**dct_sky)
    A_mu_energy = ift.SLAmplitude(**dct_energy)

    rho_mu = MfCorrelatedFieldAntares(position_space, (A_mu_sky, A_mu_energy))
    # Apply a nonlinearity

    signal = ift.exp(rho_nu) + ift.exp(rho_mu)

    # Build the line-of-sight response and define signal response

    efficiency = ift.Field(ift.makeDomain(energy_domain), val=np.ones(energy_domain.shape))
    R = ift.makeOp(efficiency)
    lamb = R(signal)

    # Specify noise
    data_space = position_space

    noise = .001
    N = ift.ScalingOperator(noise, data_space)

    # Generate mock signal and data
    mock_position = ift.from_random('normal', position_space)
    data = lamb(mock_position)
    data = np.random.poisson(data.to_global_data().astype(np.float64))
    data = ift.Field.from_global_data(data_space, data)

    # Minimization parameters
    ic_sampling = ift.GradientNormController(iteration_limit=100)
    ic_newton = ift.GradInfNormController(
        name='Newton', tol=1e-7, iteration_limit=35)
    minimizer = ift.NewtonCG(ic_newton)

    # Set up likelihood and information Hamiltonian
    likelihood = ift.PoissonianEnergy(data)(lamb)
    H = ift.StandardHamiltonian(likelihood, ic_sampling)

    initial_mean = ift.MultiField.full(H.domain, 0.)
    mean = initial_mean

    # number of samples used to estimate the KL
    N_samples = 6

    # Draw new samples to approximate the KL five times
    for i in range(5):
        # Draw new samples and minimize KL
        KL = ift.MetricGaussianKL(mean, H, N_samples)
        KL, convergence = minimizer(KL)
        mean = KL.position

        # Plot current reconstruction
        #plot = ift.Plot()
        #plot.add(signal(KL.position), title="reconstruction")
        #plot.add([A.force(KL.position), A.force(mock_position)], title="power")
        #plot.output(ny=1, ysize=6, xsize=16,
        #            name=filename.format("loop_{:02d}".format(i)))

    # Draw posterior samples
    KL = ift.MetricGaussianKL(mean, H, N_samples)
    sc = ift.StatCalculator()
    for sample in KL.samples:
        sc.add(signal(sample + KL.position))

    # Plotting
   # filename_res = filename.format("results")
    #plot = ift.Plot()
    #plot.add(sc.mean, title="Posterior Mean")
    #plot.add(ift.sqrt(sc.var), title="Posterior Standard Deviation")

    #powers = [A.force(s + KL.position) for s in KL.samples]
    #plot.add(
    #    powers + [A.force(KL.position),
    #              A.force(mock_position)],
    #    title="Sampled Posterior Power Spectrum",
    #    linewidth=[1.]*len(powers) + [3., 3.])
    #plot.output(ny=1, nx=3, xsize=24, ysize=6, name=filename_res)
    #print("Saved results as '{}'.".format(filename_res))

