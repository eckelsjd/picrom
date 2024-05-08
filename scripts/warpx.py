from pathlib import Path
import os
import tempfile

import numpy as np
from scipy import constants
import h5py
from joblib import Parallel, delayed
import tqdm
import matplotlib.pyplot as plt
import uqtils as uq

K_TO_EV = constants.physical_constants['kelvin-electron volt relationship'][0]


def process_warpx_data(parallel=True):
    """Convert warpx data into snapshot matrices of specific quantities of interest.

    ASSUMPTIONS:
    - one file per timestep, all in the same directory, with .h5 extension following openpmd standard
    - timestep files are named sequentially such that sort(names) works
    - first file is iteration=0 with time=0, last file is last iteration with time=tf
    """
    root_dir = Path('../results')
    base_path = root_dir / 'benchmark2d-1'
    data_files = sorted([f for f in os.listdir(base_path) if f.endswith('.h5')],
                        key=lambda ele: int(ele.split('_')[1].split('.')[0]))
    energy_scale = constants.c**2 * constants.m_e  # for normalized energy quantities
    Nsave = len(data_files)

    # Get important data from h5py attributes
    with h5py.File(base_path / data_files[-1], 'r') as fd:
        iter_num = str(list(fd['data'].keys())[0])
        tf = fd[f'data/{iter_num}'].attrs['time']  # final time (s)
        grid_shape = fd[f'data/{iter_num}/fields/phi'].shape
        grid_spacing = fd[f'data/{iter_num}/fields/phi'].attrs['gridSpacing']
        dtype = str(fd[f'data/{iter_num}/fields/phi'].dtype)

    # Make a dictionary to save snapshot matrices and important metadata
    with h5py.File('warpx.h5', 'a') as fd:
        if fd.get('fields') is not None:
            del fd['fields']
        iters_per_save = int(data_files[1].split('_')[1].split('.')[0])
        Niter = int(data_files[-1].split('_')[1].split('.')[0])
        dt = tf / Niter
        group = fd.create_group('fields')
        group.attrs.update({'dt': dt, 'grid_spacing': grid_spacing, 'tf': tf, 'Niter': Niter, 'Nsave': Nsave,
                            'dtype': dtype, 'iters_per_save': iters_per_save, 'coords': ['azimuthal (z)', 'axial (x)'],
                            'grid_shape': grid_shape})

    def parallel_func(idx, Te_mat, Ex_mat, Ez_mat, jx_mat, jz_mat, ni_mat):
        """Obtain QoI data from a given warp-x snapshot .h5 file (one timestep per file)"""
        with h5py.File(base_path / data_files[idx], 'r') as fd:
            iter_num = str(list(fd['data'].keys())[0])      # Only one iteration per file
            meshes = fd[f'data/{iter_num}/fields']          # Group ref for all mesh field quantities
            Ex_mat[..., idx] = meshes['E']['x'][:]          # 1MB for each qoi per timestep
            Ez_mat[..., idx] = meshes['E']['z'][:]
            jx_mat[..., idx] = meshes['j']['x'][:]
            jz_mat[..., idx] = meshes['j']['z'][:]
            ni_mat[..., idx] = meshes['rho_ions'][:] / constants.e

            # Electron energies and velocities
            ex = meshes['energy_x_electrons'][:]
            ey = meshes['energy_y_electrons'][:]
            ez = meshes['energy_z_electrons'][:]
            ux = meshes['ux_electrons'][:]
            uy = meshes['uy_electrons'][:]
            uz = meshes['uz_electrons'][:]

            # Compute electron temperature
            thermal_j = (1/2) * energy_scale * ((ex - ux**2) + (ey - uy**2) + (ez - uz**2))     # Thermal energy (J)
            Te_mat[..., idx] = (2/3) * thermal_j / constants.k * K_TO_EV                        # Temperature (eV)

    with tempfile.NamedTemporaryFile(suffix='.dat', mode='w+b') as Te_fd, \
            tempfile.NamedTemporaryFile(suffix='.dat', mode='w+b') as Ex_fd, \
            tempfile.NamedTemporaryFile(suffix='.dat', mode='w+b') as Ez_fd, \
            tempfile.NamedTemporaryFile(suffix='.dat', mode='w+b') as jx_fd, \
            tempfile.NamedTemporaryFile(suffix='.dat', mode='w+b') as jz_fd, \
            tempfile.NamedTemporaryFile(suffix='.dat', mode='w+b') as ni_fd:

        snap_shape = grid_shape + (Nsave,)
        Te_mat = np.memmap(Te_fd.name, dtype='float64', mode='r+', shape=snap_shape)
        Ex_mat = np.memmap(Ex_fd.name, dtype='float64', mode='r+', shape=snap_shape)
        Ez_mat = np.memmap(Ez_fd.name, dtype='float64', mode='r+', shape=snap_shape)
        jx_mat = np.memmap(jx_fd.name, dtype='float64', mode='r+', shape=snap_shape)
        jz_mat = np.memmap(jz_fd.name, dtype='float64', mode='r+', shape=snap_shape)
        ni_mat = np.memmap(ni_fd.name, dtype='float64', mode='r+', shape=snap_shape)

        if parallel:
            with Parallel(n_jobs=-1, verbose=10) as ppool:
                ppool(delayed(parallel_func)(idx, Te_mat, Ex_mat, Ez_mat, jx_mat, jz_mat, ni_mat) for idx in range(Nsave))
        else:
            for idx in tqdm.tqdm(range(Nsave)):
                parallel_func(idx, Te_mat, Ex_mat, Ez_mat, jx_mat, jz_mat, ni_mat)

        with h5py.File('warpx.h5', 'a') as fd:
            fd.create_dataset('fields/Te', data=Te_mat)
            fd.create_dataset('fields/Ex', data=Ex_mat)
            fd.create_dataset('fields/Ez', data=Ez_mat)
            fd.create_dataset('fields/jx', data=jx_mat)
            fd.create_dataset('fields/jz', data=jz_mat)
            fd.create_dataset('fields/ni', data=ni_mat)

            fd['fields/Te'].attrs['units'] = 'eV'
            fd['fields/Ex'].attrs['units'] = 'V/m'
            fd['fields/Ez'].attrs['units'] = 'V/m'
            fd['fields/jx'].attrs['units'] = 'A/m^2'
            fd['fields/jz'].attrs['units'] = 'A/m^2'
            fd['fields/ni'].attrs['units'] = 'm^-3'


def view_warpx_data():
    """Show plots of warp-x data"""
    with h5py.File('warpx.h5', 'r') as fd:
        Te = fd['fields/Te'][:]  # (eV)
        ni = fd['fields/ni'][:]  # (m^-3)
        Ex = fd['fields/Ex'][:]  # (V/m)
        attrs = dict(fd['fields'].attrs)

    Nx = attrs['grid_shape'][1]
    dx = attrs['grid_spacing'][1]
    dt = attrs['dt'] * attrs['iters_per_save']
    Nsave = attrs['Nsave']
    tg = np.arange(0, Nsave) * dt
    xg = np.arange(0, Nx) * dx

    x_mag = 0.75     # cm
    t_start = 16e-6  # s
    idx_start = np.argmin(np.abs(tg - t_start))
    Te_avg = np.mean(Te[..., idx_start:], axis=(0, 2))  # (Nx,)
    ni_avg = np.mean(ni[..., idx_start:], axis=(0, 2))  # (Nx,)
    Ex_avg = np.mean(Ex[..., idx_start:], axis=(0, 2))  # (Nx,)

    with plt.style.context('uqtils.default'):
        plt.rcParams.update({'text.usetex': False})
        fig, ax = plt.subplots(3, 1, sharex='col', figsize=(5, 11), layout='tight')
        ax[0].plot(xg*100, Te_avg, '-r')
        ax[0].axvline(x_mag, c='k', alpha=0.7, ls='--')
        uq.ax_default(ax[0], '', 'Electron temperature (eV)', legend=False)
        ax[1].plot(xg*100, ni_avg, '-r')
        ax[1].axvline(x_mag, c='k', alpha=0.7, ls='--')
        uq.ax_default(ax[1], '', r'Ion number density ($m^{-3}$)', legend=False)
        ax[2].plot(xg*100, Ex_avg/10**4, '-r')
        ax[2].axvline(x_mag, c='k', alpha=0.7, ls='--')
        uq.ax_default(ax[2], 'Axial distance (cm)', r'Axial electric field ($10^4$ V/m)', legend=False)
        plt.show()


if __name__ == '__main__':
    # process_warpx_data(parallel=True)
    view_warpx_data()
