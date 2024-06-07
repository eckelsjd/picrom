import h5py
import numpy as np


"""Load TURF data"""
with h5py.File('turf_ni.h5', 'r') as fd:
    turf_data = fd['fields/ni'][:]
    print(turf_data.shape)  # (N_x, N_y, N_z, N_time)

    # (Optional, but potentially recommended -- cut off transient first part of simulation)
    # idx_ss = 10  # Steady-state around 10th iteration
    # turf_data = turf_data[..., idx_ss:]


"""Load Warp-X data"""
with h5py.File('warpx_ni.h5', 'r') as fd:
    warpx_data = fd['fields/ni'][:]
    print(warpx_data.shape)  # (N_x, N_y, N_time)

    # (Optional, but potentially recommended -- cut off transient first part of simulation)
    # attrs = dict(fd['fields'].attrs)
    # N_time = warpx_data.shape[-1]
    # dt = attrs['dt']
    # t = np.arange(0, N_time) * dt
    # idx_ss = np.argmin(np.abs(t - 14e-6))  # Steady-state around 14 mu-s into simulation
    # warpx_data = warpx_data[..., idx_ss:]

# Create SVD snapshot matrices (what I currently do)
turf_snap = turf_data.reshape((-1, turf_data.shape[-1]))        # (N_states, N_time)
warpx_snap = warpx_data.reshape((-1, warpx_data.shape[-1]))     # (N_states, N_time)

# Compress to latent space (what your magic can do)
# with h5py.File('turf_ni_compressed.h5', 'a') as fd:
#     attrs = {'rank': 1, 'etc': ...}  # Add any metadata you think I might need
#     group = fd.create_group('compressed')
#     group.attrs.update(attrs)
#     turf_compressed = tensor_train(turf_data)
#     print(turf_compressed.shape)  # (N_latent, N_time)
#     fd.create_dataset('compressed/ni', data=turf_compressed)

# with h5py.File('warpx_ni_compressed.h5', 'a') as fd:
#     attrs = {'rank': 1, 'etc': ...}  # Add any metadata you think I might need
#     group = fd.create_group('compressed')
#     group.attrs.update(attrs)
#     warpx_compressed = tensor_train(warpx_data)
#     print(warpx_compressed.shape)  # (N_latent, N_time)
#     fd.create_dataset('compressed/ni', data=warpx_compressed)
