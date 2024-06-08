import numpy as np
import matplotlib.pyplot as plt
import matplotlib

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

import DaMAT as dmt  # Doruk's TT-ICE package


PRINT_NORMALIZATION = True


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
    return np.sqrt(np.sum((pred - targ)**2, axis=axis) / np.sum(targ**2, axis=axis))


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
        case 'sqrt':
            return np.sqrt(data), (c1, c2)
        case 'log10':
            return np.log10(data + 1), (c1, c2)
        case 'log-log':
            return np.log1p(np.log1p(data)), (c1, c2)
        case 'z-log':
            log_data = np.log1p(data)
            c1, c2 = np.mean(log_data), np.std(log_data)
            return (log_data - c1) / c2, (c1, c2)
        case 'sqrt-log':
            return np.log1p(np.sqrt(data)), (c1, c2)
        case 'none':
            return data, None


def denormalize(data, method='log', consts=None):
    match method:
        case 'log':
            return np.expm1(data)
        case 'zscore':
            c1, c2 = consts
            return data * c2 + c1
        case 'sqrt':
            return data ** 2
        case 'log10':
            return 10 ** data - 1
        case 'log-log':
            return np.expm1(np.expm1(data))
        case 'z-log':
            c1, c2 = consts
            return np.expm1(data * c2 + c1)
        case 'sqrt-log':
            return np.expm1(data) ** 2
        case 'none':
            return data


def warpx():
    """Compare DMD methods on Warp-X 2d ion density data"""
    # Load data
    # with h5py.File('warpx.h5', 'r') as fd:
    #     attrs = dict(fd['fields'].attrs)
    #     Nz, Nx = attrs['grid_shape']
    #     dz, dx = attrs['grid_spacing']
    #     dt = attrs['dt'] * attrs['iters_per_save']
    #     Nsave = int(attrs['Nsave'])
    #     cmap = 'bwr'
    #     qoi_full = fd[f'fields/ni'][:]  # (Nz, Nx, Nsave)
    # t = np.arange(0, Nsave) * dt            # (s)
    # x = np.arange(0, Nx) * dx * 100         # (cm)
    # z = np.arange(0, Nz) * dz * 100         # (cm)
    # idx_ss = np.argmin(np.abs(t-14e-6))     # Steady-state around 12 mu-s into simulation
    # qoi = qoi_full[..., idx_ss:]
    # t = t[idx_ss:] - t[idx_ss]
    # Nt = t.shape[0]
    # sol_exact = qoi.reshape((-1, Nt))  # (Nstates, Ntime)
    # Nstates = sol_exact.shape[0]

    # with h5py.File('warpx_ni.h5', 'a') as fd:
    #     group = fd.create_group('fields')
    #     group.attrs.update({'dt': dt, 'grid_spacing': attrs['grid_spacing'], 'grid_shape': attrs['grid_shape'],
    #                         'coords': ('N_x', 'N_y', 'N_time')})
    #     fd.create_dataset('fields/ni', data=qoi_full)

    with h5py.File('warpx_ni.h5', 'r') as fd:
        qoi = fd['fields/ni'][:]  # (Nx, Ny, Nsave)
        attrs = dict(fd['fields'].attrs)
        Nz, Nx = attrs['grid_shape']
        dz, dx = attrs['grid_spacing']
        Nsave = qoi.shape[-1]
        dt = attrs['dt']
        cmap = 'bwr'

    t = np.arange(0, Nsave) * dt
    x = np.arange(0, Nx) * dx * 100         # (cm)
    z = np.arange(0, Nz) * dz * 100         # (cm)
    idx_ss = np.argmin(np.abs(t - 14e-6))  # Steady-state around 14 mu-s into simulation
    qoi = qoi[..., idx_ss:]
    Nt = qoi.shape[-1]
    sol_exact = qoi.reshape((-1, Nt))  # (Nstates, Ntime)
    t = t[idx_ss:] - t[idx_ss]
    Nstates = sol_exact.shape[0]

    # Preprocessing
    r = 77
    pct_train = 0.5
    norm_method = 'log'
    norm1, norm2 = 'log', 'none'
    num_train = round(pct_train * Nt)
    dmd_method = 'tensor'
    eps = 0.001  # TT reconstruction accuracy
    consts = None

    match dmd_method:
        case 'exact':
            snapshot_matrix, consts = normalize(sol_exact[:, :num_train], method=norm_method)  # (Nstates, Ntrain)
            if PRINT_NORMALIZATION:
                data = snapshot_matrix
                print(f'{"Min": >10} {"Max": >10} {"Mean": >10} {"Median": >10} {"Std": >10}')
                print(f'{data.min(): >10.5f} {data.max(): >10.5f} {np.mean(data): >10.5f} {np.median(data): >10.5f} {np.std(data): >10.5f}')
        case 'tensor':
            # base_path = Path(r'C:\Users\eckel\Dropbox (University of Michigan)\compressed_turf_warpx_rom\warpx')
            base_path = Path('.')
            tcc_file = f'warpx_first150_ni_compressed_{norm1}_{norm2}_eps' + f'{eps:0.8f}'.split(".")[-1] + '.ttc'
            latent_file = f'warpx_first150_ni_latent_data_{norm1}_{norm2}_eps' + f'{eps:0.8f}'.split(".")[-1] + '.npy'
            ttObj = dmt.ttObject.loadData(str((base_path/tcc_file).resolve()))
            snapshot_matrix = np.load(base_path / latent_file)  # (Nlatent, Ntrain)
            rec = ttObj.reconstruct(snapshot_matrix).reshape(qoi.shape)
            rec_denorm = denormalize(rec, method=norm_method, consts=consts)
            targ, _ = normalize(qoi, method=norm_method)
            r = snapshot_matrix.shape[0]  # Use all the TT-latent space for DMD
            print(f'Normalized reconstruction L2 max error: {np.max(relative_error(targ, rec, axis=(0, 1)))}')
            print(f'Unnormalized reconstruction L2 max error: {np.max(relative_error(qoi, rec_denorm, axis=(0, 1)))}')

    # Exact-DMD
    dmd = DMD(svd_rank=r, exact=False)
    dmd.fit(snapshot_matrix)
    b, lamb, phi = dmd.amplitudes, dmd.eigs, dmd.modes  # (r,), (r,), and (Nstates, r)
    omega = np.log(lamb) / dt  # Continuous time eigenvalues
    sol_dmd = phi @ np.diag(b) @ np.exp(t[np.newaxis, :] * omega[:, np.newaxis])  # (Nstates, Nt)
    print(f'Imaginary sol maximum: {np.max(np.abs(np.imag(sol_dmd)))}, Real sol maximum: {np.max(np.abs(np.real(sol_dmd)))}')

    match dmd_method:
        case 'exact':
            sol_dmd = denormalize(sol_dmd.real, method=norm_method, consts=consts)
        case 'tensor':
            sol_dmd = ttObj.reconstruct(sol_dmd.real)
            sol_dmd = denormalize(sol_dmd, method=norm_method, consts=consts)

    qoi_dmd = sol_dmd.reshape((Nz, Nx, Nt))  # (Nz, Nx, Nt)
    l2_error = relative_error(qoi, qoi_dmd, axis=(0, 1))

    # Other-DMD
    # opts = dict(tol=1e-4, eps_stall=1e-10, maxiter=30, maxlam=100, lamup=1.5, verbose=True, init_lambda=0.1)
    # dmd = hankel_preprocessing(BOPDMD(svd_rank=r, num_trials=1, varpro_opts_dict=opts,
    #                                   eig_constraints={'stable'}, bag_maxfail=100), d=delay)
    # dmd.fit(snapshot_matrix, t=t[:num_train-delay+1])
    # sol_dmd = g_inv(dmd.forecast(t)[:Nstates, :].real)
    # qoi_dmd = np.swapaxes(sol_dmd.reshape((Nx, Nz, Nt)), 0, 1)  # (Nz, Nx, Nt)
    # plot_summary(dmd, x=x, y=z, t=t[:num_train], snapshots_shape=(Nx, Nz), max_sval_plot=100)

    thresh = 10
    qoi[qoi <= thresh] = np.nan
    qoi_dmd[qoi_dmd <= thresh] = np.nan  # For plotting
    vmin, vmax = np.nanmin([qoi[..., -1], qoi_dmd[..., -1]]), np.max([qoi[..., -1], qoi_dmd[..., -1]])
    imshow_args = {'extent': [0, x[-1], 0, z[-1]], 'origin': 'lower', 'vmin': vmin, 'vmax': vmax, 'cmap': cmap}
    with plt.style.context('uqtils.default'):
        with matplotlib.rc_context(rc={'font.size': 16}):
            # Ground truth final snapshot
            figsize = (7, 4)
            fig, ax = plt.subplots(figsize=figsize, layout='tight')
            im = ax.imshow(qoi[:, :, -1], **imshow_args)
            im_ratio = Nz / Nx
            cb = fig.colorbar(im, label=r'Ion density ($m^{-3}$)', fraction=0.048*im_ratio, pad=0.04)
            uq.ax_default(ax, r'Axial direction $x$ (cm)', r'Azimuthal direction $y$ (cm)', legend=False)
            ax.grid(visible=False)
            plt.show(block=False)

            # DMD final snapshot
            fig, ax = plt.subplots(figsize=figsize, layout='tight')
            im = ax.imshow(qoi_dmd[:, :, -1], **imshow_args)
            im_ratio = Nz / Nx
            cb = fig.colorbar(im, label=r'Ion density ($m^{-3}$)', fraction=0.048*im_ratio, pad=0.04)
            uq.ax_default(ax, r'Axial direction $x$ (cm)', r'Azimuthal direction $y$ (cm)', legend=False)
            ax.grid(visible=False)
            plt.show(block=False)

            # L2 error over time
            c = plt.get_cmap(cmap)(0)
            fig, ax = plt.subplots(figsize=figsize, layout='tight')
            ax.plot(t * 1e6, l2_error, '-k')
            ax.axvspan(0, t[num_train] * 1e6, alpha=0.2, color=c, label='Training period')
            ax.axvline(t[num_train] * 1e6, color=c, ls='--', lw=1)
            ax.set_yscale('log')
            uq.ax_default(ax, r'Time ($\mu$s)', r'Relative $L_2$ error', legend={'loc': 'lower right'})
            plt.show(block=False)

            # Singular value spectrum
            s = np.linalg.svd(snapshot_matrix, full_matrices=False, compute_uv=False)
            frac = s ** 2 / np.sum(s ** 2)
            r = r if isinstance(r, int) else int(np.where(np.cumsum(frac) >= r)[0][0]) + 1
            fig, ax = plt.subplots(figsize=(6, 5), layout='tight')
            ax.plot(frac, '-ok', ms=3)
            h, = ax.plot(frac[:r], 'or', ms=5, label=r'{}'.format(f'SVD rank $r={r}$'))
            ax.set_yscale('log')
            uq.ax_default(ax, 'Index', 'Fraction of total variance', legend={'loc': 'upper right'})
            plt.show(block=False)

            plt.show()


def turf_3d_plot(qoi_full, qoi_dmd, pts_full, figsize=(6, 5)):
    """Plot X,Y,Z slices of TURF data on 3d plot
    :param qoi_full: `(Nx, Ny, Nz)` the qoi to plot
    :param qoi_dmd: `(Nx, Ny, Nz)` the dmd approximation of qoi
    :param pts_full: `(Nx, Ny, Nz, 3)` the Cartesian coordinates
    :param figsize: (W, H) size of the figure (inches)
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
        cb = cb_fig.colorbar(mappable, cax=cb_ax, orientation='vertical', label=r'Ion density ($m^{-3}$)')

    return (fig, ax), (fig_dmd, ax_dmd), (cb_fig, cb_ax)


def turf():
    """Compare DMD methods on the TURF data"""
    with h5py.File('turf.h5', 'r') as fd:
        attrs = dict(fd['fields'].attrs)
        Nx, Ny, Nz = attrs['grid_shape']
        dx, dy, dz = attrs['grid_spacing']
        dt = attrs['dt'] * attrs['iters_per_save']
        Ndomain = attrs['Ndomain']
        Nsave = int(attrs['Nsave'])
        cmap = 'bwr'
        qoi = fd[f'fields/ni'][:]  # (Npts, Ndomain, Nsave)
        pts = fd['fields/coords'][:]
        permute_axes = (2, 1, 0, 3, 4)
        qoi = qoi.reshape((int(Nz / 2), int(Ny / 2), int(Nx / 2), Ndomain, Nsave))  # (Z, Y, X, Ndomain, Nsave)
        qoi = np.transpose(qoi, axes=permute_axes)
        pts = pts.reshape((int(Nz / 2), int(Ny / 2), int(Nx / 2), 3, Ndomain))      # This array comes out as (Z, Y, X, 3d, Ndomain)
        pts = np.transpose(pts, axes=permute_axes)                                  # Now in the order (X,Y,Z, 3d, Ndomain)

        # Stick the subdomains together
        subdomains = list(itertools.product([0, 1], repeat=3))
        pts_full = np.empty((Nx, Ny, Nz, 3))
        qoi_full = np.empty((Nx, Ny, Nz, Nsave), dtype=qoi.dtype)
        sub_shape = (int(Nx / 2), int(Ny / 2), int(Nz / 2))
        for i, subdomain in enumerate(subdomains):
            xs, ys, zs = [ele * sub_shape[j] for j, ele in enumerate(subdomain)]
            xe, ye, ze = xs + int(Nx / 2), ys + int(Ny / 2), zs + int(Nz / 2)  # 8 equal cube subdomains
            pts_full[xs:xe, ys:ye, zs:ze, :] = pts[..., i]
            qoi_full[xs:xe, ys:ye, zs:ze, :] = qoi[..., i, :]

    # with h5py.File('turf_ni.h5', 'a') as fd:
    #     group = fd.create_group('fields')
    #     group.attrs.update({'dt': dt, 'grid_spacing': attrs['grid_spacing'], 'grid_shape': attrs['grid_shape'],
    #                         'coords': ('N_x', 'N_y', 'N_z', 'N_time')})
    #     fd.create_dataset('fields/ni', data=qoi_full)

    # with h5py.File('turf_ni.h5', 'r') as fd:
    #     qoi = fd['fields/ni'][:]  # (Nx, Ny, Nz, Nt)
    #     attrs = dict(fd['fields'].attrs)
    #     dt = attrs['dt']
    #     dx, dy, dz = attrs['grid_spacing']
    #     Nx, Ny, Nz = attrs['grid_shape']
    #     Nsave = qoi.shape[-1]
    #     cmap = 'bwr'

    # slice_idx, slice_axis = 0, 2
    # qoi_slice = np.take(qoi_full, slice_idx, axis=slice_axis).squeeze()  # 2d slice at z=0
    idx_ss = 10
    t = np.arange(0, Nsave) * dt            # (s)
    x = np.arange(0, Nx) * dx               # (m)
    y = np.arange(0, Ny) * dy               # (m)
    z = np.arange(0, Nz) * dz               # (m)
    qoi = qoi_full[..., idx_ss:]
    Nt = qoi.shape[-1]
    sol_exact = qoi.reshape((-1, Nt))  # (Nstates, Ntime)
    t = t[idx_ss:] - t[idx_ss]
    Nstates = sol_exact.shape[0]

    # Preprocessing
    r = 50
    pct_train = 0.5
    norm_method = 'log'
    num_train = round(pct_train * Nt)
    dmd_method = 'exact'
    eps = 0.000001  # TT reconstruction accuracy
    consts = None

    match dmd_method:
        case 'exact':
            snapshot_matrix, consts = normalize(sol_exact[:, :num_train], method=norm_method)  # (Nstates, Ntrain)
            if PRINT_NORMALIZATION:
                data = snapshot_matrix
                print(f'{"Min": >10} {"Max": >10} {"Mean": >10} {"Median": >10} {"Std": >10}')
                print(f'{data.min(): >10.5f} {data.max(): >10.5f} {np.mean(data): >10.5f} {np.median(data): >10.5f} {np.std(data): >10.5f}')
        case 'tensor':
            pass

    # Exact-DMD
    dmd = DMD(svd_rank=r)
    dmd.fit(snapshot_matrix)
    b, lamb, phi = dmd.amplitudes, dmd.eigs, dmd.modes  # (r,), (r,), and (Nstates, r)
    omega = np.log(lamb) / dt  # Continuous time eigenvalues
    sol_dmd = phi @ np.diag(b) @ np.exp(t[np.newaxis, :] * omega[:, np.newaxis])  # (Nstates, Nt)
    print(f'Imaginary sol maximum: {np.max(np.abs(np.imag(sol_dmd)))}, Real sol maximum: {np.max(np.abs(np.real(sol_dmd)))}')

    match dmd_method:
        case 'exact':
            sol_dmd = denormalize(sol_dmd.real, method=norm_method, consts=consts)
        case 'tensor':
            pass
            # sol_dmd = ttObj.reconstruct(sol_dmd.real)
            # sol_dmd = denormalize(sol_dmd, method=norm_method, consts=consts)

    qoi_dmd = sol_dmd.reshape((Nx, Ny, Nz, Nt))
    l2_error = relative_error(sol_exact, sol_dmd, axis=0)

    thresh = 10
    qoi[qoi <= thresh] = np.nan
    qoi_dmd[qoi_dmd <= thresh] = np.nan  # For plotting

    slice_idx, slice_axis = 0, 2
    qoi_slice = np.take(qoi, slice_idx, axis=slice_axis).squeeze()
    dmd_slice = np.take(qoi_dmd, slice_idx, axis=slice_axis).squeeze()
    vmin, vmax = np.nanmin([qoi_slice[..., -1], dmd_slice[..., -1]]), np.nanmax([qoi_slice[..., -1], dmd_slice[..., -1]])
    imshow_args = {'extent': [0, x[-1], 0, y[-1]], 'origin': 'lower', 'vmin': vmin, 'vmax': vmax, 'cmap': cmap,
                   'norm': 'log'}
    figsize = (6, 5)
    with plt.style.context('uqtils.default'):
        # 3d final snapshots
        (t_fig, dmd_fig, cb_fig) = turf_3d_plot(qoi[..., -1], qoi_dmd[..., -1], pts_full, figsize=figsize)
        t_fig[0].savefig('turf_truth_final.pdf', bbox_inches='tight', pad_inches=0.4, format='pdf')
        dmd_fig[0].savefig(f'turf_{dmd_method}_final_r={r}.pdf', pad_inches=0.4, bbox_inches='tight', format='pdf')
        cb_fig[0].savefig(f'turf_{dmd_method}_final_r={r}_cb.pdf', bbox_inches='tight', format='pdf')
        plt.show(block=False)

        # Ground truth final snapshot (2d)
        # fig, ax = plt.subplots(figsize=figsize, layout='tight')
        # im_slice = qoi_slice[:, :, -1].T  # (Ny, Nx)
        # im = ax.imshow(im_slice, **imshow_args)
        # im.cmap.set_bad(im.cmap.get_under())
        # im_ratio = Ny / Nx
        # cb = fig.colorbar(im, label=r'Ion density ($m^{-3}$)', fraction=0.046*im_ratio, pad=0.04)
        # uq.ax_default(ax, r'Axial direction $x$ (m)', r'Radial direction $y$ (m)', legend=False)
        # ax.grid(visible=False)
        # plt.show(block=False)
        #
        # # DMD final snapshot (2d)
        # fig, ax = plt.subplots(figsize=figsize, layout='tight')
        # im_slice = dmd_slice[:, :, -1].T  # (Ny, Nx)
        # im = ax.imshow(im_slice, **imshow_args)
        # im.cmap.set_bad(im.cmap.get_under())
        # im_ratio = Ny / Nx
        # cb = fig.colorbar(im, label=r'Ion density ($m^{-3}$)', fraction=0.046*im_ratio, pad=0.04)
        # uq.ax_default(ax, r'Axial direction $x$ (m)', r'Radial direction $y$ (m)', legend=False)
        # ax.grid(visible=False)
        # plt.show(block=False)

        # L2 error over time
        c = plt.get_cmap(cmap)(0)
        fig, ax = plt.subplots(figsize=figsize, layout='tight')
        ax.plot(t*1e3, l2_error, '-k')
        ax.axvspan(0, t[num_train]*1e3, alpha=0.2, color=c, label='Training period')
        ax.axvline(t[num_train]*1e3, color=c, ls='--', lw=1)
        ax.set_yscale('log')
        uq.ax_default(ax, r'Time (ms)', r'Relative $L_2$ error', legend={'loc': 'lower right'})
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


if __name__ == '__main__':
    # tutorial()
    # diffusion_equation()
    # burgers_equation()
    # warpx()
    turf()
