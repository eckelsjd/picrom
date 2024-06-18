import itertools
from pathlib import Path
import pickle

import numpy as np
import h5py
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib


def animate_warpx(qoi_use='ni', use_dmd=False):
    with h5py.File('warpx_amrex.h5', 'r') as fd:
        attrs = dict(fd['fields'].attrs)
        Nx, Ny = attrs['grid_shape']
        dx, dy = attrs['grid_spacing']
        dt = attrs['dt'] * attrs['iters_per_save']
        Nsave = attrs['Nsave']
        t = np.arange(0, Nsave) * dt
        Nt = Nsave
        frame_skip = 5
        cmap = 'bwr'
        qoi_list = ['ni', 'jx', 'jz']
        qoi = fd[f'fields/{qoi_use}'][:]  # (Nx, Ny, Nsave)
        match qoi_use:
            case 'ni':
                norm = 'linear'
                qoi_label = r'Ion density ($m^{-3}$)'
            case 'Ez':
                norm = 'linear'
                qoi_label = r'Azimuthal electric field (V/m)'
            case 'jx':
                norm = 'linear'
                qoi_label = r'Axial current density (A/$\mathrm{m}^2$)'
            case 'jz':
                norm = 'linear'
                qoi_label = r'Azimuthal current density (A/$\mathrm{m}^2$)'

    # Override to plot dmd animation instead
    if use_dmd:
        ranks = [10, 50, 100]
        dmd_file = f'warpx_dmd_exact_r{ranks}.pkl'
        with open(Path('../results/warpx') / f'warpx_r{ranks}' / dmd_file, 'rb') as fd:
            dmd_dict = pickle.load(fd)
            qoi = dmd_dict[f'dmd_pred_r{ranks[-1]}'][..., qoi_list.index(qoi_use), :]  # Must be one of ni, jx, jz
        skip = 2
        idx_ss = np.argmin(np.abs(t - 10e-6))  # Steady-state
        sl = slice(idx_ss, None, skip)  # Only take every "skip" idx
        t = t[sl]
        Nt = t.shape[0]
        frame_skip = 2

    # Preprocessing and plot
    # thresh = 10
    qoi_plot = np.transpose(qoi, axes=(1, 0, 2))  # (Ny, Nx, Nt)
    # qoi_plot[qoi_plot < thresh] = np.nan
    vmin, vmax = np.nanmin(qoi_plot[..., -100:]), np.nanmax(qoi_plot[..., -100:])
    with matplotlib.rc_context(rc={'font.size': 15, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
                                   'text.usetex': True}):
        fig, ax = plt.subplots(figsize=(7, 4), layout='tight')
        im = ax.imshow(qoi_plot[..., 0], cmap=cmap, origin='lower', norm=norm, vmin=vmin, vmax=vmax,
                       extent=[0, qoi_plot.shape[1]*dx*100, 0, qoi.shape[1]*dy*100])
        im.cmap.set_bad((0, 0, 0, 1))
        im_ratio = Ny / Nx
        cb = fig.colorbar(im, label=qoi_label, fraction=0.048*im_ratio)
        ax.set_xlabel(r'Axial direction $x$ (cm)')
        ax.set_ylabel(r'Azimuthal direction $y$ (cm)')
        # window = 20

        def animate(i):
            idx_use = i * frame_skip
            curr_t = t[idx_use]
            im.set_data(qoi_plot[..., idx_use])
            # l_idx = max(idx_use - window, 0)
            # u_idx = min(idx_use + window, Nsave)
            # im.set_clim(np.nanmin(qoi_plot[..., l_idx:u_idx]), np.nanmax(qoi_plot[..., l_idx:u_idx]))
            # im.cmap.set_bad((0, 0, 0, 1))
            ax.set_title(r't = {} $\mu$s'.format(f'{curr_t*1e6:4.1f}'))
            return [im]

        ani = FuncAnimation(fig, animate, frames=int(Nt/frame_skip), interval=30, blit=True)
        ani.save(f'warpx_amrex-{qoi_use}-{"dmd" if use_dmd else ""}.gif')


def get_turf_slice(qoi_use, loc=0, axis: str | int = 'y', use_dmd=False):
    """Get a 2d slice of multi-domain TURF qoi at a given location and slice plane

    :param qoi_use: qoi to get a 2d slice of (ni, nn, j)
    :param loc: the point(s) at which to take the 2d slice
    :param axis: the direction along which to slice (x,y,z,t)=(0,1,2,3)
    :param use_dmd: get a slice from dmd save file instead of ground truth PIC simulation
    """
    with h5py.File('turf.h5', 'r') as fd:
        attrs = dict(fd['fields'].attrs)
        grid_shape = attrs['grid_shape']
        grid_spacing = attrs['grid_spacing']
        grid_domain = attrs['grid_domain']
        Nsave = attrs['Nsave']
        dt = attrs['dt'] * attrs['iters_per_save']
        tf = attrs['tf']
        t = np.arange(0, Nsave) * dt  # (s)
        qoi = fd[f'fields/{qoi_use}'][:]  # (Nx, Ny, Nz, Nsave)
    if isinstance(axis, str):
        axes = ['x', 'y', 'z', 't']
        axis = axes.index(axis)
    loc = np.atleast_1d(loc)

    # Override to get dmd slice for animation instead
    if use_dmd:
        idx_ss = 10
        t = t[idx_ss:]
        qoi_list = ['ni', 'nn']
        ranks = [10, 50, 100]
        dmd_file = f'turf_dmd_exact_r{ranks}.pkl'
        with open(Path('../results/turf') / f'turf_r{ranks}' / dmd_file, 'rb') as fd:
            _ = [pickle.load(fd) for i in range(2)]  # Throw out the first 2 ranks (annoying its saved this way..)
            dmd_dict = pickle.load(fd)
            qoi = dmd_dict[f'dmd_pred_r{ranks[-1]}'][..., qoi_list.index(qoi_use), :]  # Must be one of ni, nn

    # Find the indices of nearest slice locations
    domains = [(grid_domain[i, 0], grid_domain[i, 1]) for i in range(3)]  # For (x,y,z)
    grids = [np.linspace(lb + grid_spacing[i]/2, ub - grid_spacing[i]/2, grid_shape[i])
             for i, (lb, ub) in enumerate(domains)]
    grids.append(t)  # Append the time grid
    slice_idx = [int(np.argmin(np.abs(grids[axis] - l))) for l in loc]

    return t, np.squeeze(np.take(qoi, slice_idx, axis=axis), axis=axis)


def animate_turf(qoi_use='ni', loc=0, axis='y', use_dmd=False):
    """Animate a given 2d slice and location of TURF simulation data"""
    with h5py.File('turf.h5', 'r') as fd:
        attrs = dict(fd['fields'].attrs)
        grid_domain = attrs['grid_domain']
        dt = attrs['dt'] * attrs['iters_per_save']
        dz = 0.1  # m
    cmap = 'bwr'
    frame_skip = 1
    axes = ['x', 'y', 'z', 't']
    lx, ly, lz = 'Axial $x$ (m)', 'Radial $y$ (m)', 'Transverse $z$ (m)'
    xb, yb, zb = [grid_domain[i, :] for i in range(3)]
    labels = [(ly, lz), (lx, lz), (lx, ly), (lx, ly)]
    bounds = [(yb, zb), (xb, zb), (xb, yb), (xb, yb)]
    if isinstance(axis, str):
        axis = axes.index(axis)

    t, qoi = get_turf_slice(qoi_use, loc=loc, axis=axis, use_dmd=use_dmd)  # (N1, N2, N3), where (N2, N1) is the image and N3 are frames
    qoi = np.transpose(qoi, axes=(1, 0, 2))
    N2, N1, N3 = qoi.shape

    match qoi_use:
        case 'ni':
            qoi_label = r'Ion density ($m^{-3}$)'
            thresh = 10
            qoi[qoi <= thresh] = np.nan
            new_thresh = np.nanmin(qoi)
            qoi[np.isnan(qoi)] = new_thresh
            norm = 'log'
        case 'nn':
            qoi_label = r'Neutral density ($m^{-3}$)'
            thresh = 1
            qoi[qoi <= thresh] = np.nan
            new_thresh = np.nanmin(qoi)
            qoi[np.isnan(qoi)] = new_thresh
            norm = 'log'
        case 'j':
            qoi_label = r'Ion current density (A/$m^2$)'
            norm = 'linear'

    vmin, vmax = np.nanmin(qoi[..., -100:]), np.nanmax(qoi[..., -100:])
    with matplotlib.rc_context(rc={'font.size': 15, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
                                   'text.usetex': True}):
        fig, ax = plt.subplots(figsize=(5, 4), layout='tight')
        im = ax.imshow(qoi[..., 0], cmap=cmap, origin='lower', extent=[*bounds[axis][0], *bounds[axis][1]], norm=norm,
                       vmin=vmin, vmax=vmax)
        im.cmap.set_bad(im.cmap.get_under())
        im_ratio = N2 / N1
        cb = fig.colorbar(im, label=qoi_label, fraction=0.048*im_ratio)
        ax.set_xlabel(labels[axis][0])
        ax.set_ylabel(labels[axis][1])
        # window = 3

        def animate(i):
            idx_use = i * frame_skip
            curr_t = t[idx_use]
            im.set_data(qoi[..., idx_use])
            # l_idx = max(idx_use - window, 0)
            # u_idx = min(idx_use + window, N3)
            # im.set_clim(max(min_bound, np.min(qoi[..., l_idx:u_idx])), np.max(qoi[..., l_idx:u_idx]))
            # im.cmap.set_bad(im.cmap.get_under())
            ax.set_title(r't = {} ms'.format(f'{curr_t*1e3:4.1f}') if axis < 3 else r'z = {} m'.format(f'{idx_use * dz}:4.1f'))
            return [im]

        ani = FuncAnimation(fig, animate, frames=int(N3/frame_skip), interval=30, blit=True)
        ani.save(f'turf-{qoi_use}-{f"dmd" if use_dmd else ""}-axis_{axes[axis]}-loc_{loc:.1f}.gif')


if __name__ == '__main__':
    # animate_warpx(qoi_use='ni', use_dmd=False)
    animate_turf(qoi_use='nn', axis='y', loc=0, use_dmd=False)
