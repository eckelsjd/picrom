import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import pickle
import os

from pydmd import DMD, BOPDMD, EDMD
from pydmd.plotter import plot_summary
from pydmd.preprocessing import hankel_preprocessing
import scipy.linalg
import scipy.integrate
import tqdm
import h5py
import uqtils as uq
import itertools
from pathlib import Path
import argparse

import DaMAT as dmt  # Doruk's TT-ICE package


parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epsilon', type=float, help='Epsilon for TT compression', default=1e-3)
parser.add_argument('-r', '--ranks', type=int, help='Rank(s) for SVD truncation',  nargs="+", default=50)
parser.add_argument('-t', '--turf', action='store_true', default=False)
parser.add_argument('-w', '--warpx', action='store_true', default=False)
parser.add_argument('-i', '--split', type=float, help='Training/test split', default=0.5)
parser.add_argument('-c', '--compress', action='store_true', default=False, help='Run TT compression and save')
parser.add_argument('-d', '--dmd', action='store_true', default=False, help='Run DMD and save')
parser.add_argument('-p', '--plot', action='store_true', default=False, help='Plot DMD results')
args = parser.parse_args()
print(args)


def plot3d(figsize=(6, 5)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')
    # ax.axes.set_xlim3d(left=-2, right=10)
    # ax.axes.set_ylim3d(bottom=0, top=10)
    # ax.axes.set_zlim3d(bottom=0, top=10)
    ax.set_xlabel(r'Axial ($x$)', labelpad=12)
    ax.set_ylabel(r'Radial ($y$)', labelpad=12)
    ax.set_zlabel(r'Transverse ($z$)')
    ax.xaxis.set_pane_color((1, 1, 1, 0))
    ax.yaxis.set_pane_color((1, 1, 1, 0))
    ax.zaxis.set_pane_color((1, 1, 1, 0))
    ax.xaxis._axinfo["grid"]['color'] = (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] = (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] = (1,1,1,0)
    return fig, ax


def relative_error(targ, pred, axis=None):
    """Return relative L2 error between predicted and target

    :param targ: `(..., N)` the target vector
    :param pred: `(..., N)` the predicted vector
    :param axis: the axis to compute the relative error over
    """
    return np.sqrt(np.sum((pred - targ)**2, axis=axis) / np.sum(np.float64(targ)**2, axis=axis))


def tutorial():
    """Run the pyDMD tutorials"""
    # Define data and simulation parameters
    def f1(x, t):
        return 1 / np.cosh(x + 3) * np.cos(2.3 * t)

    def f2(x, t):
        return 2 / np.cosh(x) * np.tanh(x) * np.sin(2.8 * t)

    nx, nt = 65, 129
    x = np.linspace(-5, 5, nx)
    t = np.linspace(0, 4 * np.pi, nt)
    xg, tg = np.meshgrid(x, t)
    X = f1(xg, tg) + f2(xg, tg)
    X = X.T + np.random.normal(0, 0.2, size=(nx, nt))
    dt = t[1] - t[0]
    r = 4       # SVD rank approximation
    delay = 2   # Hankel time-delay

    # Fit the DMD
    # dmd = hankel_preprocessing(DMD(svd_rank=r), d=delay)
    dmd = hankel_preprocessing(BOPDMD(svd_rank=r, num_trials=0), d=delay)
    dmd.fit(X, t=t[:-delay+1])

    # Time forecasting
    # Show DMD modes
    plot_summary(dmd, x=x, d=delay)

    # Show reconstruction of snapshot matrix
    fig, ax = plt.subplots(2, 1, layout='tight', figsize=(6, 8), sharex='col')
    ax[0].imshow(dmd.reconstructed_data.real, extent=[t[0], t[-1], x[0], x[-1]], origin='lower')
    ax[1].imshow(X, extent=[t[0], t[-1], x[0], x[-1]], origin='lower')
    ax[0].set_title('Reconstructed data')
    ax[1].set_title('Original data')
    ax[1].set_xlabel('Time (s)')
    ax[0].set_ylabel('Position (m)')
    ax[1].set_ylabel('Position (m)')
    plt.show()
    # Plot relative error over time
    # Compare original and reconstructed states at end of simulation


def diffusion_equation():
    # Parameters / domain / BCs etc.
    nu = 0.1
    Nx, Nt, tf = 200, 800, 10
    x = np.linspace(-4, 4, Nx)
    t = np.linspace(0, tf, Nt)
    dx, dt = x[1] - x[0], t[1] - t[0]
    mu, std = 0, 1
    x0 = 1/(np.sqrt(2*np.pi)*std) * np.exp(-0.5*((x - mu)/std)**2)

    # Finite-difference state matrix (periodic BC)
    A = np.zeros((Nx, Nx))
    for i in range(Nx):
        A[i, i] = -2*nu / dx**2
        if i == 0:
            A[i, i+1] = nu / dx**2
            A[i, -1] = nu / dx**2
        elif i == Nx-1:
            A[i, i-1] = nu / dx**2
            A[i, 0] = nu / dx**2
        else:
            A[i, i+1] = nu / dx**2
            A[i, i-1] = nu / dx**2
    # eigval, eigvec = np.linalg.eig(A) -- DMD does not match these for some reason

    # Analytical and numerical solution
    sol_exact = np.squeeze(scipy.linalg.expm(t.reshape((Nt, 1, 1)) * A) @ x0.reshape((Nx, 1)), axis=-1).T  # (Nx, Nt)
    sol = scipy.integrate.solve_ivp(lambda t, x: A @ x, (0, tf), x0, t_eval=t)
    sol_rk45 = sol.y  # (Nx, Nt)

    # Train DMD and predict
    mu_X, std_X = np.mean(sol_exact), np.std(sol_exact)
    def g(X):
        """Transform states"""
        return (X - mu_X) / std_X
    def g_inv(g):
        """Inverse transform"""
        return g * std_X + mu_X
    r = 20
    pct_train = 0.2
    num_train = round(pct_train * Nt)
    snapshot_matrix = g(sol_exact[:, :num_train])  # (Nx, Ntrain)
    dmd = DMD(svd_rank=r)
    dmd.fit(snapshot_matrix)
    b, lamb, phi = dmd.amplitudes, dmd.eigs, dmd.modes[:Nx, :]  # (r,), (r,), and (Nx, r)
    omega = np.log(lamb) / dt  # Continuous time eigenvalues
    sol_dmd = phi @ np.diag(b) @ np.exp(t[np.newaxis, :] * omega[:, np.newaxis])  # (Nx, Nt)
    sol_dmd = g_inv(sol_dmd.real)
    l2_error = relative_error(sol_exact, sol_dmd, axis=0)

    cmap = 'viridis'
    vmin, vmax = np.nanmin([sol_dmd, sol_exact]), np.nanmax([sol_dmd, sol_exact])
    imshow_args = {'extent': [t[0], t[-1], x[0], x[-1]], 'origin': 'lower', 'vmin': vmin, 'vmax': vmax, 'cmap': cmap}
    with plt.style.context('uqtils.default'):
        # Ground truth final snapshot
        figsize = (6, 5)
        fig, ax = plt.subplots(figsize=figsize, layout='tight')
        im = ax.imshow(sol_exact, **imshow_args)
        im.cmap.set_bad(im.cmap.get_under())
        im_ratio = Nx / Nt
        cb = fig.colorbar(im, label=r'Field quanitity $u(x,t)$', fraction=0.046 * im_ratio, pad=0.04)
        uq.ax_default(ax, r'Time ($t$)', r'Position ($x$)', legend=False)
        ax.grid(visible=False)
        plt.show(block=False)

        # DMD final snapshot
        fig, ax = plt.subplots(figsize=figsize, layout='tight')
        im = ax.imshow(sol_dmd, **imshow_args)
        im.cmap.set_bad(im.cmap.get_under())
        im_ratio = Nx / Nt
        cb = fig.colorbar(im, label=r'Field quanitity $u(x,t)$', fraction=0.046 * im_ratio, pad=0.04)
        uq.ax_default(ax, r'Time ($t$)', r'Position ($x$)', legend=False)
        ax.grid(visible=False)
        plt.show(block=False)

        # L2 error over time
        c = plt.get_cmap(cmap)(0)
        fig, ax = plt.subplots(figsize=figsize, layout='tight')
        ax.plot(t, l2_error, '-k')
        ax.axvspan(0, t[num_train], alpha=0.2, color=c, label='Training period')
        ax.axvline(t[num_train], color=c, ls='--', lw=1)
        ax.set_yscale('log')
        uq.ax_default(ax, r'Time', r'Relative $L_2$ error', legend={'loc': 'lower right'})
        plt.show(block=False)

        # Singular value spectrum
        s = np.linalg.svd(snapshot_matrix, full_matrices=False, compute_uv=False)
        frac = s ** 2 / np.sum(s ** 2)
        r = r if isinstance(r, int) else int(np.where(np.cumsum(frac) >= r)[0][0]) + 1
        fig, ax = plt.subplots(figsize=figsize, layout='tight')
        ax.plot(frac, '-ok', ms=3)
        h, = ax.plot(frac[:r], 'or', ms=5, label=r'{}'.format(f'SVD rank $r={r}$'))
        ax.set_yscale('log')
        uq.ax_default(ax, 'Index', 'Fraction of total variance', legend={'loc': 'upper right'})
        plt.show(block=False)

        plt.show()


def burgers_equation():
    # Parameters / domain / BCs etc.
    nu = 0.1
    Nx, Nt, tf = 1000, 1000, 6
    x = np.linspace(-3, 3, Nx)
    t = np.linspace(0, tf, Nt)
    dx, dt = x[1] - x[0], t[1] - t[0]
    mu, std = -1, 0.5
    x0 = 1/(np.sqrt(2*np.pi)*std) * np.exp(-0.5*((x - mu)/std)**2)

    # Numerical solution
    def f(t, x):
        dxdt = np.zeros(x.shape)
        # Periodic BCs
        dxdt[0] = -x[0] * (x[1] - x[-1]) / dx + nu * (x[1] - 2*x[0] + x[-1]) / dx**2
        dxdt[-1] = -x[-1] * (x[0] - x[-2]) / dx + nu * (x[0] - 2*x[-1] + x[-2]) / dx**2
        # Everything else
        dxdt[1:-1] = -x[1:-1] * (x[2:] - x[:-2]) / dx + nu * (x[2:] - 2*x[1:-1] + x[:-2]) / dx**2
        return dxdt
    sol = scipy.integrate.solve_ivp(f, (0, tf), x0, t_eval=t)
    sol_exact = sol.y  # (Nx, Nt)

    # Analytical solution
    # def F(theta):
    #     return 1/(np.sqrt(2*np.pi)*std) * np.exp(-0.5*((theta - mu)/std)**2)  # (Gaussian IC)
    # def G(x, t, eta):
    #     return (1/(2*nu))*scipy.integrate.quad(F, 0, eta)[0] + (x-eta)**2/(4*nu*t)
    # def num_int(x, t, eta):
    #     return (x - eta)/t * np.exp(-G(x, t, eta))
    # def denom_int(x, t, eta):
    #     return np.exp(-G(x, t, eta))
    # def u(x, t):
    #     return (scipy.integrate.quad(lambda eta: num_int(x, t, eta), -np.inf, np.inf)[0] /
    #             scipy.integrate.quad(lambda eta: denom_int(x, t, eta), -np.inf, np.inf)[0])
    # sol_exact = np.zeros((Nx, Nt))
    # sol_exact[:, 0] = x0
    # for i in tqdm.tqdm(range(Nx)):
    #     for j in range(Nt-1):
    #         sol_exact[i, j+1] = u(x[i], t[j+1])

    # Preprocessing
    r = 50
    pct_train = 0.3
    norm_method = 'none'
    num_train = round(pct_train * Nt)
    snapshot_matrix, consts = normalize(sol_exact[:, :num_train], method=norm_method)  # (Nstates, Ntrain)

    # Exact-DMD
    dmd = DMD(svd_rank=r)
    dmd.fit(snapshot_matrix)
    b, lamb, phi = dmd.amplitudes, dmd.eigs, dmd.modes  # (r,), (r,), and (Nstates, r)
    omega = np.log(lamb) / dt  # Continuous time eigenvalues
    sol_dmd = phi @ np.diag(b) @ np.exp(t[np.newaxis, :] * omega[:, np.newaxis])  # (Nstates, Nt)
    print(f'Imaginary sol maximum: {np.max(np.abs(np.imag(sol_dmd)))}, Real sol maximum: {np.max(np.abs(np.real(sol_dmd)))}')
    sol_dmd = denormalize(sol_dmd.real, method=norm_method, consts=consts)
    l2_error = relative_error(sol_exact, sol_dmd, axis=0)

    cmap = 'viridis'
    vmin, vmax = np.nanmin([sol_dmd, sol_exact]), np.nanmax([sol_dmd, sol_exact])
    imshow_args = {'extent': [t[0], t[-1], x[0], x[-1]], 'origin': 'lower', 'vmin': vmin, 'vmax': vmax, 'cmap': cmap}
    with plt.style.context('uqtils.default'):
        # Ground truth final snapshot
        figsize = (6, 5)
        fig, ax = plt.subplots(figsize=figsize, layout='tight')
        im = ax.imshow(sol_exact, **imshow_args)
        im.cmap.set_bad(im.cmap.get_under())
        im_ratio = Nx / Nt
        cb = fig.colorbar(im, label=r'Field quanitity $u(x,t)$', fraction=0.046 * im_ratio, pad=0.04)
        uq.ax_default(ax, r'Time ($t$)', r'Position ($x$)', legend=False)
        ax.grid(visible=False)
        plt.show(block=False)

        # DMD final snapshot
        fig, ax = plt.subplots(figsize=figsize, layout='tight')
        im = ax.imshow(sol_dmd, **imshow_args)
        im.cmap.set_bad(im.cmap.get_under())
        im_ratio = Nx / Nt
        cb = fig.colorbar(im, label=r'Field quanitity $u(x,t)$', fraction=0.046 * im_ratio, pad=0.04)
        uq.ax_default(ax, r'Time ($t$)', r'Position ($x$)', legend=False)
        ax.grid(visible=False)
        plt.show(block=False)

        # L2 error over time
        c = plt.get_cmap(cmap)(0)
        fig, ax = plt.subplots(figsize=figsize, layout='tight')
        ax.plot(t, l2_error, '-k')
        ax.axvspan(0, t[num_train], alpha=0.2, color=c, label='Training period')
        ax.axvline(t[num_train], color=c, ls='--', lw=1)
        ax.set_yscale('log')
        uq.ax_default(ax, r'Time', r'Relative $L_2$ error', legend={'loc': 'lower right'})
        plt.show(block=False)

        # Singular value spectrum
        s = np.linalg.svd(snapshot_matrix, full_matrices=False, compute_uv=False)
        frac = s ** 2 / np.sum(s ** 2)
        r = r if isinstance(r, int) else int(np.where(np.cumsum(frac) >= r)[0][0]) + 1
        fig, ax = plt.subplots(figsize=figsize, layout='tight')
        ax.plot(frac, '-ok', ms=3)
        h, = ax.plot(frac[:r], 'or', ms=5, label=r'{}'.format(f'SVD rank $r={r}$'))
        ax.set_yscale('log')
        uq.ax_default(ax, 'Index', 'Fraction of total variance', legend={'loc': 'upper right'})
        plt.show(block=False)

        plt.show()


def normalize(data, method='log'):
    c1, c2 = None, None
    match method:
        case 'log':
            return np.log1p(data), (c1, c2)
        case 'zscore':
            c1, c2 = np.mean(data), np.std(data)
            return (data - c1) / c2, (c1, c2)
        case 'minmax':
            c1, c2 = np.min(data), np.max(data)
            return (data - c1) / (c2 - c1), (c1, c2)
        case 'sqrt':
            return np.sqrt(data), (c1, c2)
        case 'log10':
            return np.log10(data + 1), (c1, c2)
        case 'log-log':
            return np.log1p(np.log1p(data)), (c1, c2)
        case 'log-z':
            log_data = np.log1p(data)
            c1, c2 = np.mean(log_data), np.std(log_data)
            return (log_data - c1) / c2, (c1, c2)
        case 'sqrt-log':
            return np.log1p(np.sqrt(data)), (c1, c2)
        case 'log-minmax':
            log_data = np.log1p(data)
            c1, c2 = np.min(log_data), np.max(log_data)
            return (log_data - c1) / (c2 - c1), (c1, c2)
        case 'none':
            return data, None


def denormalize(data, method='log', consts=None):
    match method:
        case 'log':
            return np.expm1(data)
        case 'zscore':
            c1, c2 = consts
            return data * c2 + c1
        case 'minmax':
            c1, c2 = consts
            return data * (c2 - c1) + c1
        case 'sqrt':
            return data ** 2
        case 'log10':
            return 10 ** data - 1
        case 'log-log':
            return np.expm1(np.expm1(data))
        case 'log-z':
            c1, c2 = consts
            return np.expm1(data * c2 + c1)
        case 'sqrt-log':
            return np.expm1(data) ** 2
        case 'log-minmax':
            c1, c2 = consts
            return np.expm1(data * (c2 - c1) + c1)
        case 'none':
            return data


def warpx(tt_compress=False, run_dmd=False, plot_dmd=True, pct_train=0.5, eps=0.001, base_path=None, ranks=50):
    """Compare DMD methods on Warp-X 2d ion density data"""
    # Setup save directory
    eps_str = f"{eps:0.4f}".split(".")[-1]
    ranks = [ranks] if not isinstance(ranks, list) else ranks
    if base_path is None:
        # base_path = Path(f'warpx_eps{eps_str}')
        base_path = Path(f'warpx_r{ranks}')
        if not base_path.exists():
            os.mkdir(base_path)
    base_path = Path(base_path)
    skip = 2

    # Load data
    with h5py.File('warpx_amrex.h5', 'r') as fd:
        attrs = dict(fd['fields'].attrs)
        Nx, Nz = attrs['grid_shape']
        dx, dz = attrs['grid_spacing']
        dt = attrs['dt'] * attrs['iters_per_save']
        Nsave = int(attrs['Nsave'])
        cmap = 'bwr'
        qois = ['ni', 'jx', 'jz']
        norms = ['log', 'minmax', 'minmax']
        labels = [r'Ion density ($\mathrm{m}^{-3}$)', r'Axial current density (A/$\mathrm{m}^2$)',
                  r'Azimuthal current density (A/$\mathrm{m}^2$)']
        Nqoi = len(qois)
        t = np.arange(0, Nsave) * dt
        idx_ss = np.argmin(np.abs(t - 10e-6))  # Steady-state
        sl = slice(idx_ss, None, skip)         # Only take every "skip" idx (memory-limited)
        t = t[sl] - t[idx_ss]
        Nt = t.shape[0]
        dt = t[1] - t[0]

        sim_data = np.empty((Nx, Nz, Nqoi, Nt), dtype=np.float32)
        norm_data = np.empty((Nx, Nz, Nqoi, Nt), dtype=np.float32)
        norm_consts = []
        for i, qoi in enumerate(qois):
            sim_data[..., i, :] = fd[f'fields/{qoi}'][..., sl]  # (Nx, Nz, Nt)
            norm_data[..., i, :], consts = normalize(sim_data[..., i, :], method=norms[i])
            norm_consts.append(consts)

    x = np.arange(0, Nx) * dx * 100         # (cm)
    z = np.arange(0, Nz) * dz * 100         # (cm)
    num_train = round(pct_train * Nt)

    # Do TT-compression if necessary
    if tt_compress:
        total_time = 0
        ttObj = dmt.ttObject(norm_data[..., 0, np.newaxis], epsilon=eps)
        ttObj.changeShape([8, 8, 8, 8, 8, 4, Nqoi, 1])
        ttObj.ttDecomp(dtype=np.float64)
        total_time += ttObj.compressionTime
        print(f'Performing TT-compression for eps={eps:.5f} ...')
        print(f'{0:4d}, {ttObj.compressionRatio:09.3f}, {np.prod(ttObj.reshapedShape) / ttObj.ttCores[-1].shape[0]:12.3f}, '
              f'{ttObj.ttCores[-1].shape[0]:4d}, {ttObj.compressionTime:07.3f}, {total_time:07.3f}, {ttObj.ttRanks}')
        for simulation_idx in range(1, num_train):
            data = norm_data[..., simulation_idx, np.newaxis]
            tic = time.time()
            ttObj.ttICEstar(data, heuristicsToUse=['skip', 'occupancy'], occupancyThreshold=1)
            step_time = time.time() - tic
            total_time += step_time
            print(f'{simulation_idx:4d}, {ttObj.compressionRatio:09.3f}, {np.prod(ttObj.reshapedShape) / ttObj.ttCores[-1].shape[0]:12.3f}, ' 
                  f'{ttObj.ttCores[-1].shape[0]:4d}, {step_time:07.3f}, {total_time:07.3f}, {ttObj.ttRanks}')

        # Don't save if TT did not converge (if r is too high)
        r = ttObj.ttRanks[-2]
        if r >= num_train:
            raise SystemError(f'TT did not converge since r=num_train={r}. Try increasing epsilon...')

        ttObj.saveData(f"warpx_tt_eps{eps_str}", directory=str(base_path.resolve()), outputType="ttc")
        latent_data = np.empty((r, num_train))
        for simulation_idx in range(num_train):
            data = norm_data[..., simulation_idx, np.newaxis]
            latent_data[:, simulation_idx] = ttObj.projectTensor(data).squeeze()
        np.save(base_path / f"warpx_latent_eps{eps_str}.npy", latent_data)

        rec = ttObj.reconstruct(latent_data).reshape(norm_data[..., :num_train].shape)
        print(f'Normalized reconstruction L2 max error: {np.max(relative_error(norm_data[..., :num_train], rec, axis=(0, 1, 2)))}\n')
        print(f'TT rank r={r}')

    else:
        # Load TT-compression results from file
        print('Skipping TT for now...')
        # print('Loading TT from file...')
        # tcc_file = f'warpx_tt_eps{eps_str}.ttc'
        # latent_file = f'warpx_latent_eps{eps_str}.npy'
        # if not (base_path / tcc_file).exists() or not (base_path / latent_file).exists():
        #     raise SystemError('Must have TT compression files available to move on.')
        # ttObj = dmt.ttObject.loadData(str((base_path / tcc_file).resolve()))
        # latent_data = np.load(base_path / latent_file)  # (Nlatent, Ntrain)
        # rec = ttObj.reconstruct(latent_data).reshape(norm_data[..., :num_train].shape)
        # rec_denorm = denormalize(rec, method=norm_method, consts=consts)
        # targ, _ = normalize(qoi[..., :num_train], method=norm_method)
        # r = latent_data.shape[0]  # Use all the TT-latent space for DMD
        # print(f'Normalized reconstruction L2 max error: {np.max(relative_error(norm_data[..., :num_train], rec, axis=(0, 1, 2)))}\n')
        # print(f'TT rank r={r}')
        # print(f'Unnormalized reconstruction L2 max error: {np.max(relative_error(qoi[..., :num_train], rec_denorm, axis=(0, 1)))}')

    if run_dmd:
        snapshot_matrix = norm_data[..., :num_train].reshape((-1, num_train))  # (Nstates, Ntrain)
        data = snapshot_matrix
        print('Distribution of normalized training data:')
        print(f'{"Min": >10} {"Max": >10} {"Mean": >10} {"Median": >10} {"Std": >10}')
        print(f'{data.min(): >10.5f} {data.max(): >10.5f} {np.mean(data): >10.5f} {np.median(data): >10.5f} {np.std(data): >10.5f}\n')
        dmd_dict = {'norms': norms, 'norm_consts': norm_consts}
        for r in ranks:
            print(f'Performing exact DMD for r={r}...')  # Technically "projection" DMD, but who cares
            dmd_exact = DMD(svd_rank=r, exact=False)
            dmd_exact.fit(snapshot_matrix)
            b, lamb, phi = dmd_exact.amplitudes, dmd_exact.eigs, dmd_exact.modes  # (r,), (r,), and (Nstates, r)
            omega = np.log(lamb) / dt  # Continuous time eigenvalues
            sol_dmd = phi @ np.diag(b) @ np.exp(t[np.newaxis, :] * omega[:, np.newaxis])  # (Nstates, Nt)
            print(f'Imaginary sol maximum: {np.max(np.abs(np.imag(sol_dmd)))}, Real sol maximum: {np.max(np.abs(np.real(sol_dmd)))}')
            sol_dmd = sol_dmd.real.reshape(norm_data.shape)
            exact_dmd = sol_dmd

            # print(f'Performing TT DMD ...')
            # snapshot_matrix = latent_data
            # data = snapshot_matrix
            # print('Distribution of normalized training data:')
            # print(f'{"Min": >10} {"Max": >10} {"Mean": >10} {"Median": >10} {"Std": >10}')
            # print(f'{data.min(): >10.5f} {data.max(): >10.5f} {np.mean(data): >10.5f} {np.median(data): >10.5f} {np.std(data): >10.5f}\n')
            # dmd_tt = DMD(svd_rank=r, exact=False)
            # dmd_tt.fit(snapshot_matrix)
            # b, lamb, phi = dmd_tt.amplitudes, dmd_tt.eigs, dmd_tt.modes  # (r,), (r,), and (Nstates, r)
            # omega = np.log(lamb) / dt  # Continuous time eigenvalues
            # sol_dmd = phi @ np.diag(b) @ np.exp(t[np.newaxis, :] * omega[:, np.newaxis])  # (Nstates, Nt)
            # print(f'Imaginary sol maximum: {np.max(np.abs(np.imag(sol_dmd)))}, Real sol maximum: {np.max(np.abs(np.real(sol_dmd)))}')
            # sol_dmd = ttObj.reconstruct(sol_dmd.real).reshape(norm_data.shape)
            # tt_dmd = sol_dmd

            # Denormalize
            for i in range(Nqoi):
                exact_dmd[..., i, :] = denormalize(exact_dmd[..., i, :], method=norms[i], consts=norm_consts[i])
                # tt_dmd[..., i, :] = denormalize(tt_dmd[..., i, :], method=norms[i], consts=norm_consts[i])
            dmd_dict[f'dmd_r{r}'] = dmd_exact
            dmd_dict[f'dmd_pred_r{r}'] = exact_dmd

        with open(base_path / f'warpx_dmd_exact_r{ranks}.pkl', 'wb') as fd:
            pickle.dump(dmd_dict, fd)
        # with open(base_path / f'warpx_dmd_tt_eps{eps_str}.pkl', 'wb') as fd:
        #     pickle.dump({'dmd': dmd_tt, 'sol_dmd': tt_dmd}, fd)
    else:
        # Load DMD results from file
        exact_file = f'warpx_dmd_exact_r{ranks}.pkl'
        # tt_file = f'warpx_dmd_tt_eps{eps_str}.pkl'
        if not (base_path / exact_file).exists():
            raise SystemError('Must have DMD result files available to move on.')
        with open(base_path / exact_file, 'rb') as fd:
            dmd_dict = pickle.load(fd)
        # with open(base_path / tt_file, 'rb') as fd:
        #     tt_dmd = pickle.load(fd)['sol_dmd']

    if plot_dmd:
        with plt.style.context('uqtils.default'):
            # Ion density
            im_ratio = Nz / Nx
            qoi_idx, time_idx = 0, -1
            truth_slice = sim_data[..., qoi_idx, time_idx].T        # (Nz, Nx)
            dmd_slices, dmd_l2_errors, dmd_abs_errors = [], [], []
            for r in ranks:
                exact_dmd = dmd_dict[f'dmd_pred_r{r}']
                dmd_slice = exact_dmd[..., qoi_idx, time_idx].T     # (Nz, Nx)
                dmd_slices.append(dmd_slice)
                dmd_l2_errors.append(relative_error(sim_data[..., qoi_idx, :], exact_dmd[..., qoi_idx, :], axis=(0, 1)))
                dmd_abs_errors.append(np.abs(dmd_slice - truth_slice))
                # tt_slice = tt_dmd[..., qoi_idx, time_idx].T             # (Nz, Nx)
                # tt_l2_error = relative_error(sim_data[..., qoi_idx, :], tt_dmd[..., qoi_idx, :], axis=(0, 1))
                # tt_abs_error = np.abs(tt_slice - truth_slice)
            scale = 1.0
            vmin, vmax = np.nanmin([truth_slice] + dmd_slices) / scale, np.nanmax([truth_slice] + dmd_slices) / scale
            emin, emax = np.nanmin(dmd_abs_errors) / scale, np.nanmax(dmd_abs_errors) / scale
            imshow_args = {'extent': [0, x[-1], 0, z[-1]], 'origin': 'lower', 'aspect': 'auto'}

            # Show the first 3 ion density modes
            # nshow = 3
            # fig, ax = plt.subplots(1, nshow, sharey='row', layout='tight', figsize=(4*nshow, 3.5))
            # dmd_obj = dmd_dict[f'dmd_r{ranks[0]}']
            # mode_order = np.argsort(-np.abs(dmd_obj.amplitudes))
            # modes = dmd_obj.modes.real.reshape((Nx, Nz, Nqoi, ranks[0]))[..., mode_order]
            # modes = [modes[:, :, 0, i] for i in range(nshow)]
            # # mode_vmin, mode_vmax = np.nanmin(modes), np.nanmax(modes)
            # for i in range(nshow):
            #     ax[i].imshow(modes[i].T, **imshow_args)
            #     ax[i].set_title(f'Mode {i+1}')
            #     ax[i].set_xlabel(f'Axial $x$ (cm)')
            #     ax[i].grid(visible=False)
            #     if i == 0:
            #         ax[i].set_ylabel(f'Azimuthal $y$ (cm)')
            # fig.subplots_adjust(wspace=0, hspace=0)
            # fig.savefig(base_path / f'warpx_ni_modes.pdf', bbox_inches='tight', format='pdf')

            # Plot ion density snapshots
            fig, ax = plt.subplots(len(ranks), 3, figsize=(15, 3*len(ranks)), layout='tight', squeeze=False)
            for i, r in enumerate(ranks):
                ax[i, 0].imshow(truth_slice / scale, vmin=vmin, vmax=vmax, cmap=cmap, **imshow_args)
                im1 = ax[i, 1].imshow(dmd_slices[i] / scale, vmin=vmin, vmax=vmax, cmap=cmap, **imshow_args)
                im2 = ax[i, 2].imshow(dmd_abs_errors[i] / scale, vmin=emin, vmax=emax, cmap='viridis', **imshow_args)
                fig.colorbar(im1, fraction=0.07 * im_ratio, ax=ax[i, 1])
                fig.colorbar(im2, fraction=0.07 * im_ratio, ax=ax[i, 2])
                ax[i, 0].set_ylabel(r'{}'.format(f'$y$ (cm), $r={r}$'))
                if i == 0:
                    ax[i, 0].set_title(r'Truth ($\mathrm{m}^{-3}$)')
                    ax[i, 1].set_title(r'Exact-DMD ($\mathrm{m}^{-3}$)')
                    ax[i, 2].set_title(r'Absolute error ($\mathrm{m}^{-3}$)')

                for j in range(3):
                    ax[i, j].grid(visible=False)
                    if i == len(ranks) - 1:
                        ax[i, j].set_xlabel('$x$ (cm)')
                    else:
                        ax[i, j].set_xticklabels([])
                    if j > 0:
                        ax[i, j].set_yticklabels([])
                # im1 = ax[0, 2].imshow(tt_slice, vmin=vmin, vmax=vmax, **imshow_args)
                # im2 = ax[1, 2].imshow(tt_abs_error, vmin=emin, vmax=emax, **imshow_args)
            fig.subplots_adjust(wspace=0, hspace=0)
            fig.savefig(base_path / f'warpx_{qois[qoi_idx]}_final.pdf', bbox_inches='tight', format='pdf')

            gray = (0.5, 0.5, 0.5)
            c = plt.get_cmap('jet')(np.linspace(0, 1, len(ranks)))
            fig, ax = plt.subplots(figsize=(6, 5.5), layout='tight')
            ax.axvspan(0, t[num_train] * 1e6, alpha=0.2, color=gray, label='Training period')
            ax.axvline(t[num_train] * 1e6, color=gray, ls='--', lw=1.2)
            for i, r in enumerate(ranks):
                ax.plot(t * 1e6, dmd_l2_errors[i], ls='--', c=c[i], label=r'{}'.format(f'$r={r}$'))
                # ax.plot(t * 1e6, tt_l2_error, '--r', label='Tensor-train DMD')
            ax.set_yscale('log')
            # leg = dict(loc='upper left', ncol=1, bbox_to_anchor=(1.02, 1.025), labelspacing=0.8)
            uq.ax_default(ax, r'Time ($\mu$s)', r'Relative $L_2$ error', legend={'loc': 'lower right'})
            fig.savefig(base_path / f'warpx_{qois[qoi_idx]}_error.pdf', bbox_inches='tight', format='pdf')

            # Axial current density
            qoi_idx, time_idx = 1, -1
            truth_slice = np.mean(sim_data[..., qoi_idx, num_train:], axis=(1, 2))      # (Nx)
            # tt_slice = np.mean(tt_dmd[..., qoi_idx, num_train:], axis=(1, 2))         # (Nx)
            dmd_slices, dmd_l2_errors, dmd_abs_errors = [], [], []
            for r in ranks:
                exact_dmd = dmd_dict[f'dmd_pred_r{r}']
                dmd_slice = np.mean(exact_dmd[..., qoi_idx, num_train:], axis=(1, 2))  # (Nx)
                dmd_slices.append(dmd_slice)
                dmd_l2_errors.append(relative_error(truth_slice, dmd_slice))
                dmd_abs_errors.append(np.abs(dmd_slice - truth_slice))
            print(f'L2 error for average axuak current density--')
            print(f'Exact-DMD: {dmd_l2_errors} for ranks {ranks}')
            # print(f'TT-DMD: {tt_l2_error}')

            fig, ax = plt.subplots(figsize=(6, 5.5), layout='tight')
            ax.plot(x, truth_slice, '-k', label='Truth')
            for i, r in enumerate(ranks):
                ax.plot(x, dmd_slices[i], ls='--', c=c[i], label=r'{}'.format(f'$r={r}$'))
                # ax.plot(x, tt_slice, '--r', label='Tensor-train DMD')
            uq.ax_default(ax, r'Axial location $x$ (cm)', r'Mean axial current density (A/$\mathrm{m}^2$)',
                          legend=True)
            fig.savefig(base_path / f'warpx_{qois[qoi_idx]}_final.pdf', bbox_inches='tight', format='pdf')

            fig, ax = plt.subplots(figsize=(6, 5.5), layout='tight')
            for i, r in enumerate(ranks):
                ax.plot(x, dmd_abs_errors[i], ls='--', c=c[i], label=r'{}'.format(f'$r={r}$'))
            uq.ax_default(ax, r'Axial location $x$ (cm)', r'Absolute error (A/$\mathrm{m}^2$)', legend=True)
            fig.savefig(base_path / f'warpx_{qois[qoi_idx]}_error.pdf', bbox_inches='tight', format='pdf')

            # Azimuthal current density
            qoi_idx, time_idx = 2, -1
            truth_slice = np.mean(sim_data[..., qoi_idx, num_train:], axis=(1, 2))  # (Nx)
            # tt_slice = np.mean(tt_dmd[..., qoi_idx, num_train:], axis=(1, 2))  # (Nx)
            dmd_slices, dmd_l2_errors, dmd_abs_errors = [], [], []
            for r in ranks:
                exact_dmd = dmd_dict[f'dmd_pred_r{r}']
                dmd_slice = np.mean(exact_dmd[..., qoi_idx, num_train:], axis=(1, 2))  # (Nx)
                dmd_slices.append(dmd_slice)
                dmd_l2_errors.append(relative_error(truth_slice, dmd_slice))
                dmd_abs_errors.append(np.abs(dmd_slice - truth_slice))
            print(f'L2 error for average azimuthal current density--')
            print(f'Exact-DMD: {dmd_l2_errors} for ranks {ranks}')
            # print(f'TT-DMD: {tt_l2_error}')

            fig, ax = plt.subplots(figsize=(6, 5.5), layout='tight')
            ax.plot(x, truth_slice, '-k', label='Truth')
            for i, r in enumerate(ranks):
                ax.plot(x, dmd_slices[i], ls='--', c=c[i], label=r'{}'.format(f'$r={r}$'))
                # ax.plot(x, tt_slice, '--r', label='Tensor-train DMD')
            uq.ax_default(ax, r'Axial location $x$ (cm)', r'Mean azimuthal current density (A/$\mathrm{m}^2$)', legend=True)
            fig.savefig(base_path / f'warpx_{qois[qoi_idx]}_final.pdf', bbox_inches='tight', format='pdf')

            fig, ax = plt.subplots(figsize=(6, 5.5), layout='tight')
            for i, r in enumerate(ranks):
                ax.plot(x, dmd_abs_errors[i], ls='--', c=c[i], label=r'{}'.format(f'$r={r}$'))
            uq.ax_default(ax, r'Axial location $x$ (cm)', r'Absolute error (A/$\mathrm{m}^2$)', legend=True)
            fig.savefig(base_path / f'warpx_{qois[qoi_idx]}_error.pdf', bbox_inches='tight', format='pdf')

            # Singular value spectrum
            s = np.linalg.svd(sim_data[..., :num_train].reshape((-1, num_train)), full_matrices=False, compute_uv=False)
            s = np.float64(s)
            frac = s ** 2 / np.sum(s ** 2)
            fig, ax = plt.subplots(figsize=(6, 5), layout='tight')
            ax.plot(frac, '-ok', ms=1)
            for i in range(len(ranks)-1, -1, -1):
                r = ranks[i]
                ax.plot(frac[:r], 'o', c=c[i], ms=3, label=r'{}'.format(f'$r={r}$'))
            ax.set_yscale('log')
            uq.ax_default(ax, 'Index', 'Fraction of total variance', legend={'loc': 'upper right'})
            fig.savefig(base_path / f'warpx_svals_r{ranks}.pdf', bbox_inches='tight', format='pdf')

        # plt.show()


def turf_3d_plot(qoi_full, qoi_dmd, pts_full, figsize=(6, 5), cb_label=None):
    """Plot X,Y,Z slices of TURF data on 3d plot
    :param qoi_full: `(Nx, Ny, Nz)` the qoi to plot
    :param qoi_dmd: `(Nx, Ny, Nz)` the dmd approximation of qoi
    :param pts_full: `(Nx, Ny, Nz, 3)` the Cartesian coordinates
    :param figsize: (W, H) size of the figure (inches)
    :param cb_label: the colorbar label
    """
    with matplotlib.rc_context(rc={'xtick.minor.visible': False, 'ytick.minor.visible': False, 'xtick.major.pad': 1,
                                   'ytick.major.pad': 1}):
        thresh = 10
        qoi_full[qoi_full < thresh] = np.nan
        new_thresh = np.nanmin(qoi_full)
        qoi_dmd[qoi_dmd < new_thresh] = np.nan
        qoi_full[np.isnan(qoi_full)] = new_thresh
        qoi_dmd[np.isnan(qoi_dmd)] = new_thresh
        fig, ax = plot3d(figsize=figsize)
        fig_dmd, ax_dmd = plot3d(figsize=figsize)
        ax.view_init(elev=25, azim=135)
        ax_dmd.view_init(elev=25, azim=135)
        vmin, vmax = [], []
        slice_indices = (-1, 0, 0)
        for i, slice_idx in enumerate(slice_indices):
            qoi_slice = np.take(qoi_full, slice_idx, axis=i)
            dmd_slice = np.take(qoi_dmd, slice_idx, axis=i)
            vmin.extend([np.nanmin(qoi_slice), np.nanmin(dmd_slice)])
            vmax.append([np.nanmax(qoi_slice), np.nanmax(dmd_slice)])
        vmin, vmax = np.min(vmin), np.max(vmax)

        for i, slice_idx in enumerate(slice_indices):
            qoi_slice = np.take(qoi_full, slice_idx, axis=i)
            dmd_slice = np.take(qoi_dmd, slice_idx, axis=i)
            pts_slice = np.take(pts_full, slice_idx, axis=i)
            X, Y, Z = pts_slice[..., 0], pts_slice[..., 1], pts_slice[..., 2]
            norm = matplotlib.colors.LogNorm(vmin, vmax)
            m = plt.cm.ScalarMappable(norm=norm, cmap='bwr')
            m.set_array([])
            im = ax.plot_surface(X, Y, Z, rcount=50, ccount=50, facecolors=m.to_rgba(qoi_slice), vmin=vmin, vmax=vmax, shade=False)
            im.cmap.set_bad(im.cmap.get_under())
            im = ax_dmd.plot_surface(X, Y, Z, rcount=50, ccount=50, facecolors=m.to_rgba(dmd_slice), vmin=vmin, vmax=vmax, shade=False)
            im.cmap.set_bad(im.cmap.get_under())

        norm = matplotlib.colors.LogNorm(vmin, vmax)
        mappable = plt.cm.ScalarMappable(norm=norm, cmap='bwr')
        cb_fig, cb_ax = plt.subplots(figsize=(0.7, 5), layout='tight')
        cb = cb_fig.colorbar(mappable, cax=cb_ax, orientation='vertical', label=cb_label)

    return (fig, ax), (fig_dmd, ax_dmd), (cb_fig, cb_ax)


def turf(tt_compress=False, run_dmd=False, plot_dmd=True, pct_train=0.5, eps=0.001, base_path=None, ranks=50):
    """Compare DMD methods on the TURF data"""
    # Setup save directory
    eps_str = f"{eps:0.4f}".split(".")[-1]
    ranks = [ranks] if not isinstance(ranks, list) else ranks
    if base_path is None:
        # base_path = Path(f'turf_eps{eps_str}')
        base_path = Path(f'turf_r{ranks}')
        if not base_path.exists():
            os.mkdir(base_path)
    base_path = Path(base_path)

    with h5py.File('turf.h5', 'r') as fd:
        attrs = dict(fd['fields'].attrs)
        Nx, Ny, Nz = attrs['grid_shape']
        dx, dy, dz = attrs['grid_spacing']
        dt = attrs['dt'] * attrs['iters_per_save']
        Nsave = int(attrs['Nsave'])
        cmap = 'bwr'
        pts = fd['fields/coords'][:]
        qois = ['ni', 'nn']
        labels = [r'Ion density ($\mathrm{m}^{-3}$)', r'Neutral density ($\mathrm{m}^{-3}$)']
        norms = ['log', 'log']
        Nqoi = len(qois)
        idx_ss = 10
        t = np.arange(0, Nsave) * dt  # (s)
        t = t[idx_ss:] - t[idx_ss]
        Nt = t.shape[0]

        sim_data = np.empty((Nx, Ny, Nz, Nqoi, Nt), dtype=np.float32)
        norm_data = np.empty((Nx, Ny, Nz, Nqoi, Nt), dtype=np.float32)
        norm_consts = []
        for i, qoi in enumerate(qois):
            sim_data[..., i, :] = fd[f'fields/{qoi}'][..., idx_ss:]  # (Nx, Ny, Nz, Nt)
            norm_data[..., i, :], consts = normalize(sim_data[..., i, :], method=norms[i])
            norm_consts.append(consts)

    # x = np.arange(0, Nx) * dx               # (m)
    # y = np.arange(0, Ny) * dy               # (m)
    # z = np.arange(0, Nz) * dz               # (m)
    num_train = round(pct_train * Nt)
    snap = norm_data[..., :num_train].reshape((-1, num_train))  # (Nstates, Ntrain)
    del norm_data

    # Do TT-compression if necessary
    if tt_compress:
        total_time = 0
        ttObj = dmt.ttObject(norm_data[..., 0, np.newaxis], epsilon=eps)
        ttObj.changeShape([10, 12, 10, 10, 10, 10, Nqoi, 1])
        ttObj.ttDecomp(dtype=np.float64)
        total_time += ttObj.compressionTime
        print(f'Performing TT-compression for eps={eps:.5f} ...')
        print(
            f'{0:4d}, {ttObj.compressionRatio:09.3f}, {np.prod(ttObj.reshapedShape) / ttObj.ttCores[-1].shape[0]:12.3f}, '
            f'{ttObj.ttCores[-1].shape[0]:4d}, {ttObj.compressionTime:07.3f}, {total_time:07.3f}, {ttObj.ttRanks}')
        for simulation_idx in range(1, num_train):
            data = norm_data[..., simulation_idx, np.newaxis]
            tic = time.time()
            ttObj.ttICEstar(data, heuristicsToUse=['skip', 'occupancy'], occupancyThreshold=1)
            step_time = time.time() - tic
            total_time += step_time
            print(
                f'{simulation_idx:4d}, {ttObj.compressionRatio:09.3f}, {np.prod(ttObj.reshapedShape) / ttObj.ttCores[-1].shape[0]:12.3f}, '
                f'{ttObj.ttCores[-1].shape[0]:4d}, {step_time:07.3f}, {total_time:07.3f}, {ttObj.ttRanks}')

        # Don't save if TT did not converge (if r is too high)
        r = ttObj.ttRanks[-2]
        if r >= num_train:
            raise SystemError(f'TT did not converge since r=num_train={r}. Try increasing epsilon...')

        ttObj.saveData(f"turf_tt_eps{eps_str}", directory=str(base_path.resolve()), outputType="ttc")
        latent_data = np.empty((r, num_train))
        for simulation_idx in range(num_train):
            data = norm_data[..., simulation_idx, np.newaxis]
            latent_data[:, simulation_idx] = ttObj.projectTensor(data).squeeze()
        np.save(base_path / f"turf_latent_eps{eps_str}.npy", latent_data)

        rec = ttObj.reconstruct(latent_data).reshape(norm_data[..., :num_train].shape)
        print(f'Normalized reconstruction L2 max error: {np.max(relative_error(norm_data[..., :num_train], rec, axis=(0, 1, 2)))}\n')
        print(f'TT rank r={r}')

    else:
        # Load TT-compression results from file
        print('Skipping TT for now...')
        # print('Loading TT from file...')
        # tcc_file = f'turf_tt_eps{eps_str}.ttc'
        # latent_file = f'turf_latent_eps{eps_str}.npy'
        # if not (base_path / tcc_file).exists() or not (base_path / latent_file).exists():
        #     raise SystemError('Must have TT compression files available to move on.')
        # ttObj = dmt.ttObject.loadData(str((base_path / tcc_file).resolve()))
        # latent_data = np.load(base_path / latent_file)  # (Nlatent, Ntrain)
        # rec = ttObj.reconstruct(latent_data).reshape(norm_data[..., :num_train].shape)
        # r = latent_data.shape[0]  # Use all the TT-latent space for DMD
        # print(f'Normalized reconstruction L2 max error: {np.max(relative_error(norm_data[..., :num_train], rec, axis=(0, 1, 2)))}\n')
        # print(f'TT rank r={r}')

    if run_dmd:
        print('Distribution of normalized training data:')
        print(f'{"Min": >10} {"Max": >10} {"Mean": >10} {"Median": >10} {"Std": >10}')
        print(f'{snap.min(): >10.5f} {snap.max(): >10.5f} {np.mean(snap): >10.5f} {np.median(snap): >10.5f} {np.std(snap): >10.5f}\n')

        for r in ranks:
            dmd_dict = {'norms': norms, 'norm_consts': norm_consts}
            print(f'Performing exact DMD for r={r}...')  # Technically "projection" DMD, but who cares
            dmd_exact = DMD(svd_rank=r, exact=False)
            dmd_exact.fit(snap)
            b, lamb, phi = dmd_exact.amplitudes, dmd_exact.eigs, dmd_exact.modes  # (r,), (r,), and (Nstates, r)
            omega = np.log(lamb) / dt  # Continuous time eigenvalues
            sol_dmd = phi @ np.diag(b) @ np.exp(t[np.newaxis, :] * omega[:, np.newaxis])  # (Nstates, Nt)
            print(f'Imaginary sol maximum: {np.max(np.abs(np.imag(sol_dmd)))}, Real sol maximum: {np.max(np.abs(np.real(sol_dmd)))}')
            sol_dmd = sol_dmd.real.reshape(sim_data.shape)

            # print(f'Performing TT DMD ...')
            # snapshot_matrix = latent_data
            # data = snapshot_matrix
            # print('Distribution of normalized training data:')
            # print(f'{"Min": >10} {"Max": >10} {"Mean": >10} {"Median": >10} {"Std": >10}')
            # print(f'{data.min(): >10.5f} {data.max(): >10.5f} {np.mean(data): >10.5f} {np.median(data): >10.5f} {np.std(data): >10.5f}\n')
            # dmd_tt = DMD(svd_rank=r, exact=False)
            # dmd_tt.fit(snapshot_matrix)
            # b, lamb, phi = dmd_tt.amplitudes, dmd_tt.eigs, dmd_tt.modes  # (r,), (r,), and (Nstates, r)
            # omega = np.log(lamb) / dt  # Continuous time eigenvalues
            # sol_dmd = phi @ np.diag(b) @ np.exp(t[np.newaxis, :] * omega[:, np.newaxis])  # (Nstates, Nt)
            # print(f'Imaginary sol maximum: {np.max(np.abs(np.imag(sol_dmd)))}, Real sol maximum: {np.max(np.abs(np.real(sol_dmd)))}')
            # sol_dmd = ttObj.reconstruct(sol_dmd.real).reshape(norm_data.shape)
            # tt_dmd = sol_dmd

            # Denormalize
            for i in range(Nqoi):
                sol_dmd[..., i, :] = denormalize(sol_dmd[..., i, :], method=norms[i], consts=norm_consts[i])
                # tt_dmd[..., i, :] = denormalize(tt_dmd[..., i, :], method=norms[i], consts=norm_consts[i])
            dmd_dict[f'dmd_r{r}'] = dmd_exact
            dmd_dict[f'dmd_pred_r{r}'] = sol_dmd

            with open(base_path / f'turf_dmd_exact_r{ranks}.pkl', 'ab') as fd:
                pickle.dump(dmd_dict, fd)
            # with open(base_path / f'turf_dmd_tt_eps{eps_str}.pkl', 'wb') as fd:
            #     pickle.dump({'dmd': dmd_tt, 'sol_dmd': tt_dmd}, fd)
            del dmd_dict, dmd_exact, sol_dmd, b, phi, lamb
        del snap

    # Load DMD results from file
    exact_file = f'turf_dmd_exact_r{ranks}.pkl'
    # tt_file = f'turf_dmd_tt_eps{eps_str}.pkl'
    if not (base_path / exact_file).exists():
        raise SystemError('Must have DMD result files available to move on.')
    with open(base_path / exact_file, 'rb') as fd:
        dmd_data = []
        for i, r in enumerate(ranks):
            dmd_data.append(pickle.load(fd)[f'dmd_pred_r{r}'])
    # with open(base_path / tt_file, 'rb') as fd:
    #     tt_dmd = pickle.load(fd)['sol_dmd']

    if plot_dmd:
        with plt.style.context('uqtils.default'):
            im_ratio = Ny / Nx
            slice_idx, slice_axis, time_idx = 0, 2, -1
            for qoi_idx in range(Nqoi):
                truth_slice = np.take(sim_data[..., qoi_idx, time_idx], slice_idx, axis=slice_axis).T  # (Ny, Nx)
                dmd_slices, dmd_l2_errors, dmd_abs_errors = [], [], []
                for i, r in enumerate(ranks):
                    exact_dmd = dmd_data[i]
                    dmd_slice = np.take(exact_dmd[..., qoi_idx, time_idx], slice_idx, axis=slice_axis).T  # (Ny, Nx)
                    dmd_slices.append(dmd_slice.copy())
                    dmd_l2_errors.append(relative_error(sim_data[..., qoi_idx, :], exact_dmd[..., qoi_idx, :], axis=(0, 1, 2)))
                    dmd_abs_errors.append(np.abs(dmd_slice - truth_slice))

                thresh = 10  # For plotting really small values
                truth_slice[truth_slice <= thresh] = np.nan
                for i in range(len(ranks)):
                    dmd_slice, error_slice = dmd_slices[i], dmd_abs_errors[i]
                    dmd_slice[dmd_slice <= thresh] = np.nan
                    error_slice[error_slice <= thresh] = np.nan

                vmin, vmax = np.nanmin(truth_slice), np.nanmax([truth_slice] + dmd_slices)
                emin, emax = vmin, np.nanmax(dmd_abs_errors)
                imshow_args = {'extent': [-2, 10, 0, 10], 'norm': 'log', 'origin': 'lower', 'aspect': 'auto'}

                # Plot density snapshots
                fig, ax = plt.subplots(len(ranks), 3, figsize=(14, 3.5 * len(ranks)), layout='tight', squeeze=False)
                for i, r in enumerate(ranks):
                    im0 = ax[i, 0].imshow(truth_slice, vmin=vmin, vmax=vmax, cmap=cmap, **imshow_args)
                    im1 = ax[i, 1].imshow(dmd_slices[i], vmin=vmin, vmax=vmax, cmap=cmap, **imshow_args)
                    im2 = ax[i, 2].imshow(dmd_abs_errors[i], vmin=emin, vmax=emax, cmap='viridis', **imshow_args)
                    im0.cmap.set_bad(im0.cmap.get_under())
                    im1.cmap.set_bad(im1.cmap.get_under())
                    im2.cmap.set_bad(im2.cmap.get_under())
                    fig.colorbar(im1, fraction=0.048 * im_ratio, ax=ax[i, 1], extend='min')
                    fig.colorbar(im2, fraction=0.048 * im_ratio, ax=ax[i, 2], extend='min')
                    ax[i, 0].set_ylabel(r'{}'.format(f'$y$ (m), $r={r}$'))
                    if i == 0:
                        ax[i, 0].set_title(r'Truth ($\mathrm{m}^{-3}$)')
                        ax[i, 1].set_title(r'Exact-DMD ($\mathrm{m}^{-3}$)')
                        ax[i, 2].set_title(r'Absolute error ($\mathrm{m}^{-3}$)')

                    for j in range(3):
                        ax[i, j].grid(visible=False)
                        if i == len(ranks) - 1:
                            ax[i, j].set_xlabel('$x$ (m)')
                        else:
                            ax[i, j].set_xticklabels([])
                        if j > 0:
                            ax[i, j].set_yticklabels([])

                    # im1 = ax[0, 2].imshow(tt_slice, vmin=vmin, vmax=vmax, **imshow_args)
                    # im2 = ax[1, 2].imshow(tt_abs_error, vmin=emin, vmax=emax, **imshow_args)
                fig.subplots_adjust(wspace=0, hspace=0)
                fig.savefig(base_path / f'turf_{qois[qoi_idx]}_final.pdf', bbox_inches='tight', format='pdf')

                # L2 error over time
                gray = (0.5, 0.5, 0.5)
                c = plt.get_cmap('jet')(np.linspace(0, 1, len(ranks)))
                fig, ax = plt.subplots(figsize=(6, 5.5), layout='tight')
                ax.axvspan(0, t[num_train] * 1e3, alpha=0.2, color=gray, label='Training period')
                ax.axvline(t[num_train] * 1e3, color=gray, ls='--', lw=1.2)
                for i, r in enumerate(ranks):
                    ax.plot(t * 1e3, dmd_l2_errors[i], ls='--', c=c[i], label=r'{}'.format(f'$r={r}$'))
                    # ax.plot(t * 1e6, tt_l2_error, '--r', label='Tensor-train DMD')
                ax.set_yscale('log')
                uq.ax_default(ax, r'Time ($\mu$s)', r'Relative $L_2$ error', legend={'loc': 'lower right'})
                fig.savefig(base_path / f'turf_{qois[qoi_idx]}_error.pdf', bbox_inches='tight', format='pdf')

                # 3d final snapshots (only for highest rank)
                figsize = (6, 5)
                exact_dmd = dmd_data[-1]
                (t_fig, dmd_fig, cb_fig) = turf_3d_plot(sim_data[..., qoi_idx, time_idx], exact_dmd[..., qoi_idx, time_idx],
                                                        pts, figsize=figsize, cb_label=labels[qoi_idx])
                t_fig[0].savefig(base_path / f'turf_{qois[qoi_idx]}_truth_3d_r{ranks[-1]}.pdf', bbox_inches='tight', pad_inches=0.4, format='pdf')
                dmd_fig[0].savefig(base_path / f'turf_{qois[qoi_idx]}_dmd_3d_r{ranks[-1]}.pdf', pad_inches=0.4, bbox_inches='tight', format='pdf')
                cb_fig[0].savefig(base_path / f'turf_{qois[qoi_idx]}_cb_r{ranks[-1]}.pdf', bbox_inches='tight', format='pdf')

            # Singular value spectrum
            s = np.linalg.svd(sim_data[..., :num_train].reshape((-1, num_train)), full_matrices=False, compute_uv=False)
            s = np.float64(s)
            frac = s ** 2 / np.sum(s ** 2)
            fig, ax = plt.subplots(figsize=(6, 5), layout='tight')
            ax.plot(frac, '-ok', ms=1)
            for i in range(len(ranks)-1, -1, -1):
                r = ranks[i]
                ax.plot(frac[:r], 'o', c=c[i], ms=3, label=r'{}'.format(f'$r={r}$'))
            ax.set_yscale('log')
            uq.ax_default(ax, 'Index', 'Fraction of total variance', legend={'loc': 'upper right'})
            fig.savefig(base_path / f'turf_svals_r{ranks}.pdf', bbox_inches='tight', format='pdf')

        # plt.show()


if __name__ == '__main__':
    # tutorial()
    # diffusion_equation()
    # burgers_equation()
    if args.warpx:
        warpx(tt_compress=args.compress, run_dmd=args.dmd, plot_dmd=args.plot, pct_train=args.split, eps=args.epsilon, ranks=args.ranks)
    if args.turf:
        turf(tt_compress=args.compress, run_dmd=args.dmd, plot_dmd=args.plot, pct_train=args.split, eps=args.epsilon, ranks=args.ranks)
