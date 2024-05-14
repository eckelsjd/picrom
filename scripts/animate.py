import itertools

import numpy as np
import h5py
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib


def animate_warpx(qoi_use='ni'):
    with h5py.File('warpx.h5', 'r') as fd:
        attrs = dict(fd['fields'].attrs)
        Nz, Nx = attrs['grid_shape']
        dz, dx = attrs['grid_spacing']
        dt = attrs['dt'] * attrs['iters_per_save']
        Nsave = attrs['Nsave']
        cmap = 'bwr'
        qoi = fd[f'fields/{qoi_use}'][:]  # (Nz, Nx, Nsave)
        match qoi_use:
            case 'ni':
                scale = 1e18
                qoi_label = r'Ion density ($10^{18} m^{-3}$)'
            case 'Te':
                scale = 1
                qoi_label = r'Electron temperature (eV)'
            case 'Ex':
                scale = 1e4
                qoi_label = r'Axial electric field ($10^4$ V/m)'
                qoi = qoi[:, 10:-10, :]
            case 'Ez':
                scale = 1e4
                qoi_label = r'Azimuthal electric field ($10^4$ V/m)'

    # t = np.arange(0, Nsave) * dt
    # x = np.arange(0, Nx) * dx * 100
    # z = np.arange(0, Nz) * dz * 100
    # xg, zg = np.meshgrid(x, z)  # (Nz, Nx)

    with matplotlib.rc_context(rc={'font.size': 15, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
                                   'text.usetex': True}):
        fig, ax = plt.subplots(figsize=(7, 4), layout='tight')
        im = ax.imshow(qoi[..., 0], cmap=cmap, origin='lower', extent=[0, qoi.shape[1]*dx*100, 0, qoi.shape[0]*dz*100])
        im_ratio = Nz / Nx
        cb = fig.colorbar(im, label=qoi_label, fraction=0.046*im_ratio, pad=0.04, format='%5.1f')
        ax.set_xlabel(r'Axial direction $z$ (cm)')
        ax.set_ylabel(r'Azimuthal direction $\theta$ (cm)')
        skip = 2
        window = 20

        def animate(i):
            idx_use = i * skip
            curr_t = idx_use * dt
            im.set_data(qoi[..., idx_use] / scale)
            l_idx = max(idx_use - window, 0)
            u_idx = min(idx_use + window, Nsave)
            im.set_clim(np.min(qoi[..., l_idx:u_idx] / scale), np.max(qoi[..., l_idx:u_idx] / scale))
            ax.set_title(r't = {} $\mu$s'.format(f'{curr_t*1e6:4.1f}'))
            return [im]

        ani = FuncAnimation(fig, animate, frames=int(Nsave/skip), interval=30, blit=True)
        ani.save(f'warpx-{qoi_use}.gif')


def get_turf_slice(qoi, loc=0, axis: str | int = 'y'):
    """Get a 2d slice of multi-domain TURF qoi at a given location and slice plane

    :param qoi: qoi to get a 2d slice of (ni, nn, j)
    :param loc: the point(s) at which to take the 2d slice
    :param axis: the direction along which to slice (x,y,z,t)=(0,1,2,3)
    """
    with h5py.File('turf.h5', 'r') as fd:
        attrs = dict(fd['fields'].attrs)
        grid_shape = attrs['grid_shape']
        Nx, Ny, Nz = grid_shape
        grid_spacing = attrs['grid_spacing']
        grid_domain = attrs['grid_domain']
        Ndomain = attrs['Ndomain']
        Nsave = attrs['Nsave']
        dt = attrs['dt'] * attrs['iters_per_save']
        tf = attrs['tf']
        qoi = fd[f'fields/{qoi}'][:]  # (Npts, Ndomain, Nsave)
    if isinstance(axis, str):
        axes = ['x', 'y', 'z', 't']
        axis = axes.index(axis)
    loc = np.atleast_1d(loc)
    permute_axes = (2, 1, 0, 3, 4)
    qoi = qoi.reshape((int(Nz/2), int(Ny/2), int(Nx/2), Ndomain, Nsave))    # (Z, Y, X, Ndomain, Nsave)
    qoi = np.transpose(qoi, axes=permute_axes)                              # (X, Y, Z, Ndomain, Nsave)

    # Stick the subdomains together
    subdomains = list(itertools.product([0, 1], repeat=3))
    qoi_full = np.empty((Nx, Ny, Nz, Nsave), dtype=qoi.dtype)
    sub_shape = (int(Nx/2), int(Ny/2), int(Nz/2))
    for i, subdomain in enumerate(subdomains):
        xs, ys, zs = [ele * sub_shape[j] for j, ele in enumerate(subdomain)]
        xe, ye, ze = xs+int(Nx/2), ys+int(Ny/2), zs+int(Nz/2)  # 8 equal cube subdomains
        qoi_full[xs:xe, ys:ye, zs:ze] = qoi[..., i, :]

    # Find the indices of nearest slice locations
    domains = [(grid_domain[i, 0], grid_domain[i, 1]) for i in range(3)]  # For (x,y,z)
    grids = [np.linspace(lb + grid_spacing[i]/2, ub - grid_spacing[i]/2, grid_shape[i])
             for i, (lb, ub) in enumerate(domains)]
    grids.append(np.linspace(0, tf, Nsave))  # Append the time grid
    slice_idx = [int(np.argmin(np.abs(grids[axis] - l))) for l in loc]

    return np.squeeze(np.take(qoi_full, slice_idx, axis=axis), axis=axis)


def animate_turf(qoi_use='ni', loc=0, axis='y'):
    """Animate a given 2d slice and location of TURF simulation data"""
    with h5py.File('turf.h5', 'r') as fd:
        attrs = dict(fd['fields'].attrs)
        grid_domain = attrs['grid_domain']
        dt = attrs['dt'] * attrs['iters_per_save']
        dz = 0.1  # m
    cmap = 'viridis'
    axes = ['x', 'y', 'z', 't']
    lx, ly, lz = 'Axial $x$ (m)', 'Radial $y$ (m)', 'Transverse $z$ (m)'
    xb, yb, zb = [grid_domain[i, :] for i in range(3)]
    labels = [(ly, lz), (lx, lz), (lx, ly), (lx, ly)]
    bounds = [(yb, zb), (xb, zb), (xb, yb), (xb, yb)]
    if isinstance(axis, str):
        axis = axes.index(axis)

    qoi = get_turf_slice(qoi_use, loc=loc, axis=axis)  # (N1, N2, N3), where (N2, N1) is the image and N3 are frames
    qoi = np.transpose(qoi, axes=(1, 0, 2))
    N2, N1, N3 = qoi.shape

    match qoi_use:
        case 'ni':
            scale = 1
            qoi_label = r'Ion density (log $m^{-3}$)'
            qoi = np.log10(qoi)
        case 'nn':
            scale = 1
            qoi_label = r'Neutral density (log m^{-3}$)'
            qoi = np.log10(qoi)
        case 'j':
            scale = 1
            qoi_label = r'Ion current density (A/$m^2$)'

    with matplotlib.rc_context(rc={'font.size': 15, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
                                   'text.usetex': True}):
        fig, ax = plt.subplots(figsize=(5, 4), layout='tight')
        im = ax.imshow(qoi[..., 0] / scale, cmap=cmap, origin='lower', extent=[*bounds[axis][0], *bounds[axis][1]])
        im_ratio = N2 / N1
        cb = fig.colorbar(im, label=qoi_label, fraction=0.046*im_ratio, pad=0.04, format='%5.1f')
        ax.set_xlabel(labels[axis][0])
        ax.set_ylabel(labels[axis][1])
        window = 2
        skip = 1

        def animate(i):
            idx_use = i * skip
            curr_t = idx_use * dt
            im.set_data(qoi[..., idx_use] / scale)
            l_idx = max(idx_use - window, 0)
            u_idx = min(idx_use + window, N3)
            im.set_clim(np.min(qoi[..., l_idx:u_idx] / scale), np.max(qoi[..., l_idx:u_idx] / scale))
            ax.set_title(r't = {} ms'.format(f'{curr_t*1e3:4.1f}') if axis < 3 else r'z = {} m'.format(f'{idx_use * dz}:4.1f'))
            return [im]

        ani = FuncAnimation(fig, animate, frames=int(N3/skip), interval=30, blit=True)
        ani.save(f'turf-{qoi_use}-axis_{axes[axis]}-loc_{loc:.1f}.gif')


if __name__ == '__main__':
    # animate_warpx(qoi_use='ni')
    animate_turf(qoi_use='ni')
