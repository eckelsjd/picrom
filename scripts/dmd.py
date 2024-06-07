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
    r, delay = 8, 4
    pct_train = 0.2
    num_train = round(pct_train * Nt)
    snapshot_matrix = g(sol_exact[:, :num_train])  # (Nx, Ntrain)
    dmd = hankel_preprocessing(DMD(svd_rank=r), d=delay)
    dmd.fit(snapshot_matrix)
    b, lamb, phi = dmd.amplitudes, dmd.eigs, dmd.modes[:Nx, :]  # (r,), (r,), and (Nx, r)
    omega = np.log(lamb) / dt  # Continuous time eigenvalues
    sol_dmd = phi @ np.diag(b) @ np.exp(t[np.newaxis, :] * omega[:, np.newaxis])  # (Nx, Nt)
    sol_dmd = g_inv(sol_dmd.real)

    bopdmd = hankel_preprocessing(BOPDMD(svd_rank=r, num_trials=0), d=delay)
    bopdmd.fit(snapshot_matrix, t=t[:num_train-delay+1])
    sol_bopdmd = g_inv(bopdmd.forecast(t)[:Nx, :].real)

    # Plot results
    imshow_args = {'extent': [t[0], t[-1], x[0], x[-1]], 'origin': 'lower', 'vmin': np.min([sol_exact, sol_rk45, sol_dmd, sol_bopdmd]),
                   'vmax': np.max([sol_exact, sol_rk45, sol_dmd, sol_bopdmd]), 'cmap': 'viridis'}
    fig, ax = plt.subplots(3, 2, layout='tight', figsize=(10, 12), sharex='col', sharey='row')
    ax[0, 0].imshow(sol_exact, **imshow_args)
    ax[0, 1].imshow(sol_rk45, **imshow_args)
    ax[1, 0].imshow(sol_dmd, **imshow_args)
    ax[1, 1].imshow(sol_bopdmd, **imshow_args)
    ax[1, 0].axvline(t[num_train], c='r', ls='--', lw=2)
    ax[1, 1].axvline(t[num_train], c='r', ls='--', lw=2)
    ax[0, 0].set_title('Analytical')
    ax[0, 1].set_title('RK45 numerical')
    ax[1, 0].set_title('Exact DMD')
    ax[1, 1].set_title('BOP DMD')

    ax[2, 0].plot(t[:num_train], relative_error(sol_exact[:, :num_train], sol_dmd[:, :num_train], axis=0), '-k')
    ax[2, 0].plot(t[num_train:], relative_error(sol_exact[:, num_train:], sol_dmd[:, num_train:], axis=0), '-r')
    ax[2, 0].axvline(t[num_train], c='r', ls='--', lw=2)
    ax[2, 0].set_yscale('log')
    ax[2, 1].plot(t[:num_train], relative_error(sol_exact[:, :num_train], sol_bopdmd[:, :num_train], axis=0), '-k')
    ax[2, 1].plot(t[num_train:], relative_error(sol_exact[:, num_train:], sol_bopdmd[:, num_train:], axis=0), '-r')
    ax[2, 1].axvline(t[num_train], c='r', ls='--', lw=2)
    ax[2, 1].set_yscale('log')

    ax[0, 0].set_ylabel('Position (m)')
    ax[1, 0].set_ylabel('Position (m)')
    ax[2, 0].set_ylabel('Relative $L_2$ error')
    ax[2, 0].set_xlabel('Time (s)')
    ax[2, 1].set_xlabel('Time (s)')
    plt.show()


def burgers_equation():
    # Parameters / domain / BCs etc.
    nu = 0.1
    Nx, Nt, tf = 200, 1000, 10
    x = np.linspace(-4, 4, Nx)
    t = np.linspace(0, tf, Nt)
    dx, dt = x[1] - x[0], t[1] - t[0]
    mu, std = -2, 0.5
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
    sol_rk45 = sol.y  # (Nx, Nt)
    sol_exact = sol_rk45

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

    # Train DMD and predict
    mu_X, std_X = np.mean(sol_exact), np.std(sol_exact)
    def g(X):
        """Transform states"""
        return X
        # return (X - mu_X) / std_X
    def g_inv(g):
        """Inverse transform"""
        return g
        # return g * std_X + mu_X
    r, delay = 8, 4
    pct_train = 0.5
    num_train = round(pct_train * Nt)
    snapshot_matrix = g(sol_exact[:, :num_train])  # (Nx, Ntrain)
    dmd = hankel_preprocessing(DMD(svd_rank=r), d=delay)
    dmd.fit(snapshot_matrix)
    b, lamb, phi = dmd.amplitudes, dmd.eigs, dmd.modes[:Nx, :]  # (r,), (r,), and (Nx, r)
    omega = np.log(lamb) / dt  # Continuous time eigenvalues
    sol_dmd = phi @ np.diag(b) @ np.exp(t[np.newaxis, :] * omega[:, np.newaxis])  # (Nx, Nt)
    sol_dmd = g_inv(sol_dmd.real)

    edmd = EDMD(svd_rank=r, kernel_metric='polynomial', kernel_params=dict(degree=3))
    edmd.fit(snapshot_matrix)
    b, lamb, phi = edmd.amplitudes, edmd.eigs, edmd.modes
    omega = np.log(lamb) / dt
    sol_edmd = phi @ np.diag(b) @ np.exp(t[np.newaxis, :] * omega[:, np.newaxis])  # (Nx, Nt)
    sol_edmd = g_inv(sol_edmd.real)
    # bopdmd = hankel_preprocessing(BOPDMD(svd_rank=r, num_trials=10), d=delay)
    # bopdmd.fit(snapshot_matrix, t=t[:num_train-delay+1])
    # sol_bopdmd = g_inv(bopdmd.forecast(t)[:Nx, :].real)

    # Plot results
    imshow_args = {'extent': [t[0], t[-1], x[0], x[-1]], 'origin': 'lower', 'vmin': np.min([sol_exact, sol_rk45, sol_dmd, sol_edmd]),
                   'vmax': np.max([sol_exact, sol_rk45, sol_dmd, sol_edmd]), 'cmap': 'viridis'}
    fig, ax = plt.subplots(3, 2, layout='tight', figsize=(10, 12), sharex='col', sharey='row')
    ax[0, 0].imshow(sol_exact, **imshow_args)
    ax[0, 1].imshow(sol_rk45, **imshow_args)
    ax[1, 0].imshow(sol_dmd, **imshow_args)
    ax[1, 1].imshow(sol_edmd, **imshow_args)
    ax[1, 0].axvline(t[num_train], c='r', ls='--', lw=2)
    ax[1, 1].axvline(t[num_train], c='r', ls='--', lw=2)
    ax[0, 0].set_title('Analytical')
    ax[0, 1].set_title('RK45 numerical')
    ax[1, 0].set_title('Exact DMD')
    ax[1, 1].set_title('Extended DMD')

    ax[2, 0].plot(t[:num_train], relative_error(sol_exact[:, :num_train], sol_dmd[:, :num_train], axis=0), '-k')
    ax[2, 0].plot(t[num_train:], relative_error(sol_exact[:, num_train:], sol_dmd[:, num_train:], axis=0), '-r')
    ax[2, 0].axvline(t[num_train], c='r', ls='--', lw=2)
    ax[2, 0].set_yscale('log')
    ax[2, 1].plot(t[:num_train], relative_error(sol_exact[:, :num_train], sol_edmd[:, :num_train], axis=0), '-k')
    ax[2, 1].plot(t[num_train:], relative_error(sol_exact[:, num_train:], sol_edmd[:, num_train:], axis=0), '-r')
    ax[2, 1].axvline(t[num_train], c='r', ls='--', lw=2)
    ax[2, 1].set_yscale('log')

    ax[0, 0].set_ylabel('Position (m)')
    ax[1, 0].set_ylabel('Position (m)')
    ax[2, 0].set_ylabel('Relative $L_2$ error')
    ax[2, 0].set_xlabel('Time (s)')
    ax[2, 1].set_xlabel('Time (s)')
    plt.show()


def warpx():
    with h5py.File('warpx.h5', 'r') as fd:
        attrs = dict(fd['fields'].attrs)
        Nz, Nx = attrs['grid_shape']
        dz, dx = attrs['grid_spacing']
        dt = attrs['dt'] * attrs['iters_per_save']
        Nsave = int(attrs['Nsave'])
        cmap = 'bwr'
        qoi_full = fd[f'fields/ni'][:]  # (Nz, Nx, Nsave)
    t = np.arange(0, Nsave) * dt            # (s)
    x = np.arange(0, Nx) * dx * 100         # (cm)
    z = np.arange(0, Nz) * dz * 100         # (cm)
    idx_ss = np.argmin(np.abs(t-14e-6))     # Steady-state around 12 mu-s into simulation
    qoi = qoi_full[..., idx_ss:]
    t = t[idx_ss:] - t[idx_ss]
    Nt = t.shape[0]
    sol_exact = np.swapaxes(qoi, 0, 1).reshape((-1, Nt))  # (Nstates, Ntime)
    Nstates = sol_exact.shape[0]

    # with h5py.File('warpx_ni.h5', 'a') as fd:
    #     group = fd.create_group('fields')
    #     group.attrs.update({'dt': dt, 'grid_spacing': attrs['grid_spacing'], 'grid_shape': attrs['grid_shape'],
    #                         'coords': ('N_x', 'N_y', 'N_time')})
    #     fd.create_dataset('fields/ni', data=qoi_full)

    # Preprocessing and fit DMD
    thresh = 1
    idx = sol_exact < thresh
    sol_exact[idx] = np.nan
    new_thresh = np.nanmin(sol_exact)
    sol_exact[idx] = new_thresh
    mu_X, std_X = np.mean(np.log(sol_exact)), np.std(np.log(sol_exact))
    g = lambda X: (np.log(X) - mu_X) / std_X    # Transform
    g_inv = lambda X: np.exp(X * std_X + mu_X)  # Inverse transform
    r, delay = 10, 1
    pct_train = 0.25
    num_train = round(pct_train * Nt)
    snapshot_matrix = g(sol_exact[:, :num_train])  # (Nstates, Ntrain)
    dmd = hankel_preprocessing(DMD(svd_rank=r), d=delay)
    dmd.fit(snapshot_matrix)
    b, lamb, phi = dmd.amplitudes, dmd.eigs, dmd.modes[:Nstates, :]  # (r,), (r,), and (Nstates, r)
    print(lamb.shape)
    omega = np.log(lamb) / dt  # Continuous time eigenvalues
    sol_dmd = phi @ np.diag(b) @ np.exp(t[np.newaxis, :] * omega[:, np.newaxis])  # (Nstates, Nt)
    sol_dmd = g_inv(sol_dmd.real)
    qoi_dmd = np.swapaxes(sol_dmd.reshape((Nx, Nz, Nt)), 0, 1)  # (Nz, Nx, Nt)
    l2_error = relative_error(sol_exact, sol_dmd, axis=0)

    # opts = dict(tol=1e-4, eps_stall=1e-10, maxiter=30, maxlam=100, lamup=1.5, verbose=True, init_lambda=0.1)
    # dmd = hankel_preprocessing(BOPDMD(svd_rank=r, num_trials=1, varpro_opts_dict=opts,
    #                                   eig_constraints={'stable'}, bag_maxfail=100), d=delay)
    # dmd.fit(snapshot_matrix, t=t[:num_train-delay+1])
    # sol_dmd = g_inv(dmd.forecast(t)[:Nstates, :].real)
    # qoi_dmd = np.swapaxes(sol_dmd.reshape((Nx, Nz, Nt)), 0, 1)  # (Nz, Nx, Nt)

    plot_summary(dmd, x=x, y=z, t=t[:num_train-delay+1], d=delay, snapshots_shape=(Nx, Nz), max_sval_plot=100)

    qoi[qoi <= new_thresh] = np.nan
    qoi_dmd[qoi_dmd <= new_thresh] = np.nan
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
            uq.ax_default(ax, r'Axial direction $z$ (cm)', r'Azimuthal direction $\theta$ (cm)', legend=False)
            ax.grid(visible=False)

            # DMD final snapshot
            fig, ax = plt.subplots(figsize=figsize, layout='tight')
            im = ax.imshow(qoi_dmd[:, :, -1], **imshow_args)
            im_ratio = Nz / Nx
            cb = fig.colorbar(im, label=r'Ion density ($m^{-3}$)', fraction=0.048*im_ratio, pad=0.04)
            uq.ax_default(ax, r'Axial direction $z$ (cm)', r'Azimuthal direction $\theta$ (cm)', legend=False)
            ax.grid(visible=False)

            # L2 error over time
            c = plt.get_cmap(cmap)(0)
            fig, ax = plt.subplots(figsize=figsize, layout='tight')
            ax.plot(t * 1e6, l2_error, '-k')
            ax.axvspan(0, t[num_train] * 1e6, alpha=0.2, color=c, label='Training period')
            ax.axvline(t[num_train] * 1e6, color=c, ls='--', lw=1)
            ax.set_yscale('log')
            uq.ax_default(ax, r'Time (ms)', r'Relative $L_2$ error', legend={'loc': 'lower right'})

            # Singular value spectrum
            s = np.linalg.svd(snapshot_matrix, full_matrices=False, compute_uv=False)
            frac = s ** 2 / np.sum(s ** 2)
            r = r if isinstance(r, int) else int(np.where(np.cumsum(frac) >= r)[0][0]) + 1
            fig, ax = plt.subplots(figsize=(6, 5), layout='tight')
            ax.plot(frac, '-ok', ms=3)
            h, = ax.plot(frac[:r], 'or', ms=5, label=r'{}'.format(f'SVD rank $r={r}$'))
            ax.set_yscale('log')
            uq.ax_default(ax, 'Index', 'Fraction of total variance', legend={'loc': 'upper right'})

            plt.show()


def turf():
    with h5py.File('turf.h5', 'r') as fd:
        attrs = dict(fd['fields'].attrs)
        Nx, Ny, Nz = attrs['grid_shape']
        dx, dy, dz = attrs['grid_spacing']
        dt = attrs['dt'] * attrs['iters_per_save']
        Ndomain = attrs['Ndomain']
        Nsave = int(attrs['Nsave'])
        cmap = 'bwr'
        qoi = fd[f'fields/ni'][:]  # (Npts, Ndomain, Nsave)
        permute_axes = (2, 1, 0, 3, 4)
        qoi = qoi.reshape((int(Nz / 2), int(Ny / 2), int(Nx / 2), Ndomain, Nsave))  # (Z, Y, X, Ndomain, Nsave)
        qoi = np.transpose(qoi, axes=permute_axes)

        # Stick the subdomains together
        subdomains = list(itertools.product([0, 1], repeat=3))
        qoi_full = np.empty((Nx, Ny, Nz, Nsave), dtype=qoi.dtype)
        sub_shape = (int(Nx / 2), int(Ny / 2), int(Nz / 2))
        for i, subdomain in enumerate(subdomains):
            xs, ys, zs = [ele * sub_shape[j] for j, ele in enumerate(subdomain)]
            xe, ye, ze = xs + int(Nx / 2), ys + int(Ny / 2), zs + int(Nz / 2)  # 8 equal cube subdomains
            qoi_full[xs:xe, ys:ye, zs:ze, :] = qoi[..., i, :]

    # with h5py.File('turf_ni.h5', 'a') as fd:
    #     group = fd.create_group('fields')
    #     group.attrs.update({'dt': dt, 'grid_spacing': attrs['grid_spacing'], 'grid_shape': attrs['grid_shape'],
    #                         'coords': ('N_x', 'N_y', 'N_z', 'N_time')})
    #     fd.create_dataset('fields/ni', data=qoi_full)

    slice_idx, slice_axis, idx_ss = 0, 2, 10
    qoi_slice = np.take(qoi_full, slice_idx, axis=slice_axis).squeeze()  # 2d slice at z=0
    t = np.arange(0, Nsave) * dt            # (s)
    x = np.arange(0, Nx) * dx               # (m)
    y = np.arange(0, Ny) * dy               # (m)
    z = np.arange(0, Nz) * dz               # (m)
    sol_exact = qoi_slice.reshape((-1, Nsave))  # (Nstates, Ntime)
    sol_exact = sol_exact[:, idx_ss:]
    t = t[idx_ss:] - t[idx_ss]
    Nt = t.shape[0]
    qoi_slice = qoi_slice[..., idx_ss:]
    Nstates = sol_exact.shape[0]

    # Preprocessing and fit DMD
    thresh = 1
    idx = sol_exact < thresh
    sol_exact[idx] = np.nan
    new_thresh = np.nanmin(sol_exact)
    sol_exact[idx] = new_thresh
    mu_X, std_X = 0, np.std(np.log(sol_exact))
    g = lambda X: (np.log(X) - mu_X) / std_X    # Transform
    g_inv = lambda X: np.exp(X * std_X + mu_X)  # Inverse transform
    r, delay = 10, 1
    pct_train = 0.25
    num_train = round(pct_train * Nt)
    snapshot_matrix = g(sol_exact[:, :num_train])  # (Nstates, Ntrain)
    dmd = hankel_preprocessing(DMD(svd_rank=r), d=delay)
    dmd.fit(snapshot_matrix)
    b, lamb, phi = dmd.amplitudes, dmd.eigs, dmd.modes[:Nstates, :]  # (r,), (r,), and (Nstates, r)
    print(lamb.shape)
    omega = np.log(lamb) / dt  # Continuous time eigenvalues
    sol_dmd = phi @ np.diag(b) @ np.exp(t[np.newaxis, :] * omega[:, np.newaxis])  # (Nstates, Nt)
    sol_dmd = g_inv(sol_dmd.real)
    qoi_dmd = sol_dmd.reshape((Nx, Ny, Nt))
    l2_error = relative_error(sol_exact, sol_dmd, axis=0)

    # opts = dict(tol=1e-4, eps_stall=1e-10, maxiter=30, maxlam=100, lamup=1.5, verbose=True, init_lambda=0.1)
    # dmd = hankel_preprocessing(BOPDMD(svd_rank=r, num_trials=1, varpro_opts_dict=opts,
    #                                   eig_constraints={'stable'}, bag_maxfail=100), d=delay)
    # dmd.fit(snapshot_matrix, t=t[:num_train-delay+1])
    # sol_dmd = g_inv(dmd.forecast(t)[:Nstates, :].real)
    # qoi_dmd = np.swapaxes(sol_dmd.reshape((Nx, Nz, Nt)), 0, 1)  # (Nz, Nx, Nt)

    plot_summary(dmd, x=x, y=y, snapshots_shape=(Nx, Ny), t=t[:num_train-delay+1], d=delay, max_sval_plot=100)

    qoi_slice[qoi_slice <= new_thresh] = np.nan
    qoi_dmd[qoi_dmd <= new_thresh] = np.nan
    vmin, vmax = np.nanmin([qoi_slice[..., -1], qoi_dmd[..., -1]]), np.nanmax([qoi_slice[..., -1], qoi_dmd[..., -1]])
    imshow_args = {'extent': [0, x[-1], 0, y[-1]], 'origin': 'lower', 'vmin': vmin, 'vmax': vmax, 'cmap': cmap,
                   'norm': 'log'}
    with plt.style.context('uqtils.default'):
        # Ground truth final snapshot
        figsize = (6, 5)
        fig, ax = plt.subplots(figsize=figsize, layout='tight')
        im_slice = qoi_slice[:, :, -1].T  # (Ny, Nx)
        im = ax.imshow(im_slice, **imshow_args)
        im.cmap.set_bad(im.cmap.get_under())
        im_ratio = Ny / Nx
        cb = fig.colorbar(im, label=r'Ion density ($m^{-3}$)', fraction=0.046*im_ratio, pad=0.04)
        uq.ax_default(ax, r'Axial direction $z$ (m)', r'Radial direction $r$ (m)', legend=False)
        ax.grid(visible=False)

        # DMD final snapshot
        fig, ax = plt.subplots(figsize=figsize, layout='tight')
        im_slice = qoi_dmd[:, :, -1].T  # (Ny, Nx)
        im = ax.imshow(im_slice, **imshow_args)
        im.cmap.set_bad(im.cmap.get_under())
        im_ratio = Ny / Nx
        cb = fig.colorbar(im, label=r'Ion density ($m^{-3}$)', fraction=0.046*im_ratio, pad=0.04)
        uq.ax_default(ax, r'Axial direction $z$ (m)', r'Radial direction $r$ (m)', legend=False)
        ax.grid(visible=False)

        # L2 error over time
        c = 'b'
        fig, ax = plt.subplots(figsize=figsize, layout='tight')
        ax.plot(t*1e3, l2_error, '-k')
        ax.axvspan(0, t[num_train]*1e3, alpha=0.2, color=c, label='Training period')
        ax.axvline(t[num_train]*1e3, color=c, ls='--', lw=1)
        ax.set_yscale('log')
        uq.ax_default(ax, r'Time (ms)', r'Relative $L_2$ error', legend={'loc': 'lower right'})

        # Singular value spectrum
        s = np.linalg.svd(snapshot_matrix, full_matrices=False, compute_uv=False)
        frac = s ** 2 / np.sum(s ** 2)
        r = r if isinstance(r, int) else int(np.where(np.cumsum(frac) >= r)[0][0]) + 1
        fig, ax = plt.subplots(figsize=figsize, layout='tight')
        ax.plot(frac, '-ok', ms=3)
        h, = ax.plot(frac[:r], 'or', ms=5, label=r'{}'.format(f'SVD rank $r={r}$'))
        ax.set_yscale('log')
        uq.ax_default(ax, 'Index', 'Fraction of total variance', legend={'loc': 'upper right'})

        plt.show()


if __name__ == '__main__':
    # tutorial()
    # diffusion_equation()
    # burgers_equation()
    warpx()
    # turf()
