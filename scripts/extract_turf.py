import os
from pathlib import Path
import tempfile

import numpy as np
import h5py
import vtk
from vtk.numpy_interface import dataset_adapter as dsa
import tqdm
from joblib import Parallel, delayed


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
            filepath = str(base_path / data_files[t_idx + d_idx])
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
    process_turf_data(parallel=True)
    # view_turf_data()
