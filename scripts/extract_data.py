import os
from pathlib import Path
import tempfile

import numpy as np
import h5py
import vtk
from scipy import constants
from vtk.numpy_interface import dataset_adapter as dsa
import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import uqtils as uq
import yt

K_TO_EV = constants.physical_constants['kelvin-electron volt relationship'][0]


def process_warpx_amrex(parallel=True):
    """Convert warpx (amrex) data into snapshot matrices of specific quantities of interest."""
    root_dir = Path('../results/steady_10us_checkpoint')
    base_path = root_dir / 'diags'
    plotfile_key = 'hall'
    data_dirs = sorted([f for f in os.listdir(base_path) if (base_path/f).is_dir() and f.startswith(plotfile_key)],
                       key=lambda ele: int(ele.split(plotfile_key)[1]))
    Nsave = len(data_dirs)

    # Get important metadata
    ds = yt.load(base_path / data_dirs[0])
    iters_per_save = int(data_dirs[1].split('hall')[1]) - int(data_dirs[0].split('hall')[1])
    left_edge = ds.domain_left_edge
    dims = ds.domain_dimensions
    grid_shape = tuple(dims[:-1])
    cov_grid = ds.covering_grid(level=0, left_edge=left_edge, dims=dims)
    grid_spacing = (np.max(cov_grid['dx'].to_ndarray()), np.max(cov_grid['dy'].to_ndarray()))

    # Save snapshot matrices and metadata
    with h5py.File('warpx_amrex.h5', 'a') as fd:
        if fd.get('fields') is not None:
            del fd['fields']
        dt = 5e-12
        group = fd.create_group('fields')
        group.attrs.update({'dt': dt, 'grid_spacing': grid_spacing, 'Nsave': Nsave, 'iters_per_save': iters_per_save,
                            'coords': ['Axial (x)', 'Azimuthal (y)'], 'grid_shape': grid_shape})

    def parallel_func(idx, Ey_mat, ni_mat):
        """Obtain QoI data from a given warp-x plotfile (amrex) directory (one timestep per directory)"""
        print(f'Processing idx {idx} -- file {data_dirs[idx]}')
        ds = yt.load(base_path / data_dirs[idx])
        cov_grid = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)
        Ey_mat[..., idx] = cov_grid['Ey'].to_ndarray().squeeze()
        ni_mat[..., idx] = - cov_grid['rho_ions'].to_ndarray().squeeze() / constants.e  # Amrex has (-) for ions?

    with (tempfile.NamedTemporaryFile(suffix='.dat', mode='w+b') as Ey_fd,
          tempfile.NamedTemporaryFile(suffix='.dat', mode='w+b') as ni_fd):

        snap_shape = grid_shape + (Nsave,)
        Ey_mat = np.memmap(Ey_fd.name, dtype='float64', mode='r+', shape=snap_shape)
        ni_mat = np.memmap(ni_fd.name, dtype='float64', mode='r+', shape=snap_shape)

        if parallel:
            with Parallel(n_jobs=-1, verbose=10) as ppool:
                ppool(delayed(parallel_func)(idx, Ey_mat, ni_mat) for idx in range(Nsave))
        else:
            for idx in tqdm.tqdm(range(Nsave)):
                parallel_func(idx, Ey_mat, ni_mat)

        with h5py.File('warpx_amrex.h5', 'a') as fd:
            fd.create_dataset('fields/Ey', data=Ey_mat)
            fd.create_dataset('fields/ni', data=ni_mat)
            fd['fields/Ey'].attrs['units'] = 'V/m'
            fd['fields/ni'].attrs['units'] = 'm**(-3)'


def process_warpx_data(parallel=True):
    """Convert warpx (openpmd) data into snapshot matrices of specific quantities of interest.

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

    # Save snapshot matrices and important metadata
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


def verify_warpx_data():
    """Show plots of warp-x data to compare to benchmark paper (Charoy 2020)"""
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


def process_turf_data(parallel=True):
    """Convert multi-domain TURF data (.vts structured mesh) to snapshot matrices."""
    root_dir = Path('../results')
    base_path = root_dir / 'Plot3DExp3p2'
    data_files = sorted([f for f in os.listdir(base_path) if f.endswith('.vts')],
                        key=lambda ele: (int(ele.split('_')[1]), ele.split('.')[0][-5:]))

    # Gather metadata
    Ndomain = 8                                 # Number of subdomains in computational domain
    Nsave = int(len(data_files) / Ndomain)      # Number of saved iterations
    iters_per_save = int(data_files[Ndomain].split('_')[1])
    Niter = int(data_files[-1].split('_')[1])   # Total number of iterations
    dtype = 'float32'
    coords = ['x', 'y', 'z']
    grid_spacing = (0.1, 0.1, 0.1)                          # m
    grid_shape = (120, 100, 100)                            # N_cells
    sub_shape = tuple([int(ele/2) for ele in grid_shape])   # Subdomain shape
    grid_domain = ((-2, 10), (0, 10), (0, 10))              # m
    dt = 5e-6                                               # s / iteration
    tf = dt * Niter                                         # final time (s)

    # Get x,y,z cell centers of all subdomain points (use first iteration files)
    pts = np.empty((int(np.prod(sub_shape)), 3, Ndomain), dtype=np.float32)
    for i in range(Ndomain):
        filepath = str(base_path / data_files[i])
        reader = vtk.vtkXMLStructuredGridReader()
        reader.SetFileName(filepath)
        reader.Update()
        struct_grid = reader.GetOutput()
        centers = vtk.vtkCellCenters()
        centers.SetInputData(struct_grid)
        centers.Update()
        pts[..., i] = np.array(dsa.WrapDataObject(centers.GetOutput()).Points)
        del reader
        del struct_grid

    # Save snapshot matrices and important metadata
    with h5py.File('turf.h5', 'a') as fd:
        if fd.get('fields') is not None:
            del fd['fields']
        group = fd.create_group('fields')
        group.attrs.update({'dt': dt, 'grid_spacing': grid_spacing, 'tf': tf, 'Niter': Niter, 'Nsave': Nsave,
                            'dtype': dtype, 'iters_per_save': iters_per_save, 'coords': coords,
                            'grid_shape': grid_shape, 'Ndomain': Ndomain, 'grid_domain': grid_domain})
        fd.create_dataset('fields/coords', data=pts)
        fd['fields/coords'].attrs.update({'axes': ['N_points', 'N_dim', 'N_subdomains'], 'type': 'cartesian'})

    def parallel_func(t_idx, nn_mat, ni_mat, j_mat):
        """Obtain QoI data from TURF .vts files at a single timestep"""
        for d_idx in range(Ndomain):
            filepath = str(base_path / data_files[t_idx*Ndomain + d_idx])
            reader = vtk.vtkXMLStructuredGridReader()
            reader.SetFileName(filepath)
            reader.Update()
            struct_grid = reader.GetOutput()
            cell_data = struct_grid.GetCellData()
            nn_mat[:, d_idx, t_idx] = np.array(dsa.vtkDataArrayToVTKArray(cell_data.GetArray("n_Xe@g")))         # Neutrals (Ncells,)
            ni_mat[:, d_idx, t_idx] = (np.array(dsa.vtkDataArrayToVTKArray(cell_data.GetArray("n_Xe+@g"))) +     # Add all Xe+ species
                                       np.array(dsa.vtkDataArrayToVTKArray(cell_data.GetArray("n_Xe+Prim"))) +
                                       np.array(dsa.vtkDataArrayToVTKArray(cell_data.GetArray("n_Xe+CEX"))))
            j_mat[:, d_idx, t_idx] = np.array(dsa.vtkDataArrayToVTKArray(cell_data.GetArray("J_Ion")))
            # Te_mat[:, d_idx, t_idx] = np.array(dsa.vtkDataArrayToVTKArray(cell_data.GetArray("TeeV")))

            del reader
            del struct_grid

    with tempfile.NamedTemporaryFile(suffix='.dat', mode='w+b') as n_fd, \
            tempfile.NamedTemporaryFile(suffix='.dat', mode='w+b') as ni_fd, \
            tempfile.NamedTemporaryFile(suffix='.dat', mode='w+b') as j_fd:

        snap_shape = (int(np.prod(sub_shape)), Ndomain, Nsave)
        nn_mat = np.memmap(n_fd, dtype='float32', mode='r+', shape=snap_shape)      # Neutral density (m^-3)
        ni_mat = np.memmap(ni_fd, dtype='float32', mode='r+', shape=snap_shape)     # Xe+ ion density (m^-3)
        j_mat = np.memmap(j_fd, dtype='float32', mode='r+', shape=snap_shape)       # Ion current density (A/m^2)

        if parallel:
            with Parallel(n_jobs=-1, verbose=10) as ppool:
                ppool(delayed(parallel_func)(idx, nn_mat, ni_mat, j_mat) for idx in range(Nsave))
        else:
            for idx in tqdm.tqdm(range(Nsave)):
                parallel_func(idx, nn_mat, ni_mat, j_mat)

        with h5py.File('turf.h5', 'a') as fd:
            fd.create_dataset('fields/nn', data=nn_mat)
            fd.create_dataset('fields/ni', data=ni_mat)
            fd.create_dataset('fields/j', data=j_mat)
            fd['fields/nn'].attrs['units'] = 'm^-3'
            fd['fields/ni'].attrs['units'] = 'm^-3'
            fd['fields/j'].attrs['units'] = 'A/m^2'


if __name__ == "__main__":
    process_warpx_amrex(parallel=True)
    # process_warx_data(parallel=True)
    # process_turf_data(parallel=True)
