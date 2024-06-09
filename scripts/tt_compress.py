import h5py
import time
import argparse
from pathlib import Path

import numpy as np
import DaMAT as dmt

from turfUtils import normalize
np.set_printoptions(suppress=False,linewidth=np.nan)

DATA_PATH = Path('.')
HEURISTICS = ['skip', 'occupancy']
# NORMALIZATION_1 = 'log'
# NORMALIZATION_2 = 'none'
COMPRESS = True

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epsilon', type=float, help='Epsilon', default=1e-1)
parser.add_argument('-n', '--normalization', nargs=2, help='Normalization method', default=['log', 'none'])
parser.add_argument('-p', '--print_normalization', action='store_true', default=False)
parser.add_argument('-t', '--turf', action='store_true', default=False)
parser.add_argument('-w', '--warpx', action='store_true', default=False)
parser.add_argument('-i', '--split', type=float, help='Training/test split', default=0.5)

args = parser.parse_args()

EPS = args.epsilon
NORMALIZATION_1, NORMALIZATION_2 = args.normalization

print(args)
print(NORMALIZATION_1, NORMALIZATION_2)

PRINT_NORMALIZATION = args.print_normalization
TURF = args.turf
WARPX = args.warpx

total_time = 0

if TURF:
    """Load TURF data"""
    idx_ss = 10  # Steady-state around 10th iteration
    with h5py.File(DATA_PATH / 'turf_ni.h5', 'r') as fd:
        turf_data = fd['fields/ni'][:]
        num_train = round((turf_data.shape[-1] - idx_ss) * args.split)
        train_idx = idx_ss + num_train
        # print(turf_data.dtype)
        print(turf_data.shape)  # (N_x, N_y, N_z, N_time)
        data = turf_data[..., idx_ss][...,None]
        # print(data.min(), data.max(), np.mean(data), np.median(data), np.std(data), " ".join(map(str:05.2f, np.quantile(data, [0.1,0.2,0.5,0.75,0.9]))))
        if PRINT_NORMALIZATION:
            print(f'{idx_ss:3d} {data.min():8.6f}, {data.max():25.3f}, {np.mean(data):25.5f}, {np.median(data):20.5f}, {np.std(data):20.5f}, {" ".join(map(lambda x: f"{x:15.4f}", np.quantile(data, [0.01,0.1,0.2,0.5,0.75,0.9])))}')
        data, nc1, nc2 = normalize(data, NORMALIZATION_1)
        data, nc3 ,nc4 = normalize(data, NORMALIZATION_2)
        if PRINT_NORMALIZATION:
            print(f'{idx_ss:3d} {data.min():8.6f}, {data.max():25.3f}, {np.mean(data):25.5f}, {np.median(data):20.5f}, {np.std(data):20.5f}, {" ".join(map(lambda x: f"{x:15.4f}", np.quantile(data, [0.01,0.1,0.2,0.5,0.75,0.9])))}')
            # print(data.min(), data.max(), np.mean(data), np.median(data), np.std(data), np.quantile(data, [0.1,0.2,0.5,0.75,0.9]))
            print(nc1,nc2,nc3,nc4)
            print()
        # quit()
        if COMPRESS:
            dataset = dmt.ttObject(
                data,
                epsilon=EPS,
            )
            dataset.normalizationMethods = [NORMALIZATION_1, NORMALIZATION_2]
            dataset.normalizationConstants = [[nc1, nc2, nc3, nc4]]
            dataset.changeShape([10,12,10,10,10,10,1])
            # dataset.changeShape([4,5,6,5,5,4,5,5,4,1])
            # dataset.changeShape([120,100,100,1])
            dataset.ttDecomp(dtype=np.float64)
            total_time += dataset.compressionTime
            # print(idx_ss, dataset.compressionRatio, np.prod(dataset.reshapedShape)/dataset.ttCores[-1].shape[0], dataset.ttCores[-1].shape[0], dataset.compressionTime, total_time, dataset.ttRanks)
            print(f'{idx_ss:03d}, {dataset.compressionRatio:09.3f}, {np.prod(dataset.reshapedShape)/dataset.ttCores[-1].shape[0]:12.3f}, {dataset.ttCores[-1].shape[0]:4d}, {dataset.compressionTime:07.3f}, {total_time:07.3f}, {dataset.ttRanks}')
        for snapshot_idx in range(idx_ss+1, train_idx):
            # print(snapshot_idx, turf_data[..., snapshot_idx].min(), turf_data[..., snapshot_idx].max(), np.mean(turf_data[..., snapshot_idx]), np.median(turf_data[..., snapshot_idx]), np.std(turf_data[..., snapshot_idx]))
            data = turf_data[..., snapshot_idx][...,None]
            # print(data.min(), data.max(), np.mean(data), np.median(data), np.std(data), np.quantile(data, [0.1,0.2,0.5,0.75,0.9]))
            if PRINT_NORMALIZATION:
                print(f'{snapshot_idx:3d} {data.min():8.6f}, {data.max():25.3f}, {np.mean(data):25.5f}, {np.median(data):20.5f}, {np.std(data):20.5f}, {" ".join(map(lambda x: f"{x:15.4f}", np.quantile(data, [0.01,0.1,0.2,0.5,0.75,0.9])))}')
            data, nc1, nc2 = normalize(data, NORMALIZATION_1)
            data, nc3 ,nc4 = normalize(data, NORMALIZATION_2)
            dataset.normalizationConstants.append([nc1, nc2, nc3, nc4])
            if PRINT_NORMALIZATION:
                print(f'{snapshot_idx:3d} {data.min():8.6f}, {data.max():25.3f}, {np.mean(data):25.5f}, {np.median(data):20.5f}, {np.std(data):20.5f}, {" ".join(map(lambda x: f"{x:15.4f}", np.quantile(data, [0.01,0.1,0.2,0.5,0.75,0.9])))}')
                # print(data.min(), data.max(), np.mean(data), np.median(data), np.std(data), np.quantile(data, [0.1,0.2,0.5,0.75,0.9]))
                print(nc1,nc2,nc3,nc4)
                print()
            if COMPRESS:
                tic = time.time()
                dataset.ttICEstar(
                    data,
                    heuristicsToUse=HEURISTICS,
                    occupancyThreshold=1,
                )
                step_time = time.time()-tic
                total_time += step_time
                print(f'{snapshot_idx:03d}, {dataset.compressionRatio:09.3f}, {np.prod(dataset.reshapedShape)/dataset.ttCores[-1].shape[0]:12.3f}, {dataset.ttCores[-1].shape[0]:4d}, {step_time:07.3f}, {total_time:07.3f}, {dataset.ttRanks}')
                # print(snapshot_idx, dataset.compressionRatio, np.prod(dataset.reshapedShape)/dataset.ttCores[-1].shape[0], dataset.ttCores[-1].shape[0], step_time, total_time, dataset.ttRanks)
        
        if COMPRESS:
            dataset.saveData(
                f"turf_ni_compressed_{NORMALIZATION_1}_{NORMALIZATION_2}_eps"+f"{EPS:0.8f}".split(".")[-1],
                directory = "../results/turf/",
                outputType = "ttc",
            )
            latent_data = np.zeros((dataset.ttRanks[-2], turf_data.shape[-1] - idx_ss))
            for idx, snapshot_idx in enumerate(range(idx_ss, turf_data.shape[-1])):
                data = turf_data[..., snapshot_idx][...,None]
                data, nc1, nc2 = normalize(data, NORMALIZATION_1)
                data, nc3 ,nc4 = normalize(data, NORMALIZATION_2)
                # print(latent_data[... , idx].shape , dataset.projectTensor(data).shape)
                latent_data[..., idx] = dataset.projectTensor(data).squeeze()
            np.save(
                f"../results/turf/turf_ni_latent_data_{NORMALIZATION_1}_{NORMALIZATION_2}_eps"+f"{EPS:0.8f}".split(".")[-1]+".npy", latent_data
            )

if WARPX:
    """Load Warp-X data"""
    with h5py.File(DATA_PATH / 'warpx_ni.h5', 'r') as fd:
        warpx_data = fd['fields/ni'][:]
        print(warpx_data.shape)  # (N_x, N_y, N_time)
        # print(warpx_data.dtype)
        attrs = dict(fd['fields'].attrs)
        N_time = warpx_data.shape[-1]
        dt = attrs['dt']
        t = np.arange(0, N_time) * dt
        idx_ss = np.argmin(np.abs(t - 14e-6))  # Steady-state around 14 mu-s into simulation
        train_idx = idx_ss + round((N_time - idx_ss) * args.split)
        # idx_ss = 0
        # print(idx_ss,dt,t[695:705], 700*dt)

        data = warpx_data[..., idx_ss][...,None]
        if PRINT_NORMALIZATION:
            print(f'{idx_ss:4d} {data.min():8.6f}, {data.max():25.3f}, {np.mean(data):25.5f}, {np.median(data):20.5f}, {np.std(data):20.5f}, {" ".join(map(lambda x: f"{x:15.4f}", np.quantile(data, [0.01,0.1,0.2,0.5,0.75,0.9])))}')
        data, nc1, nc2 = normalize(data, NORMALIZATION_1)
        data, nc3 ,nc4 = normalize(data, NORMALIZATION_2)
        if PRINT_NORMALIZATION:
            print(f'{idx_ss:4d} {data.min():8.6f}, {data.max():25.3f}, {np.mean(data):25.5f}, {np.median(data):20.5f}, {np.std(data):20.5f}, {" ".join(map(lambda x: f"{x:15.4f}", np.quantile(data, [0.01,0.1,0.2,0.5,0.75,0.9])))}')
            # print(data.min(), data.max(), np.mean(data), np.median(data), np.std(data), np.quantile(data, [0.1,0.2,0.5,0.75,0.9]))
            print(nc1,nc2,nc3,nc4)
            print()
        # quit()
        if COMPRESS:
            dataset = dmt.ttObject(
                data,
                epsilon= EPS,
            )
            dataset.normalizationMethods = [NORMALIZATION_1, NORMALIZATION_2]
            dataset.normalizationConstants = [[nc1, nc2, nc3, nc4]]
            dataset.changeShape([4,8,8,8,8,8,1])
            dataset.ttDecomp(dtype=np.float64)
            total_time += dataset.compressionTime
            print(f'{idx_ss:4d}, {dataset.compressionRatio:09.3f}, {np.prod(dataset.reshapedShape)/dataset.ttCores[-1].shape[0]:12.3f}, {dataset.ttCores[-1].shape[0]:4d}, {dataset.compressionTime:07.3f}, {total_time:07.3f}, {dataset.ttRanks}')
        for simulation_idx in range(idx_ss+1, train_idx):
            data = warpx_data[..., simulation_idx][...,None]
            if PRINT_NORMALIZATION:
                print(f'{simulation_idx:4d} {data.min():8.6f}, {data.max():25.3f}, {np.mean(data):25.5f}, {np.median(data):20.5f}, {np.std(data):20.5f}, {" ".join(map(lambda x: f"{x:15.4f}", np.quantile(data, [0.01,0.1,0.2,0.5,0.75,0.9])))}')
            data, nc1, nc2 = normalize(data, NORMALIZATION_1)
            data, nc3 ,nc4 = normalize(data, NORMALIZATION_2)
            if PRINT_NORMALIZATION:
                print(f'{simulation_idx:4d} {data.min():8.6f}, {data.max():25.3f}, {np.mean(data):25.5f}, {np.median(data):20.5f}, {np.std(data):20.5f}, {" ".join(map(lambda x: f"{x:15.4f}", np.quantile(data, [0.01,0.1,0.2,0.5,0.75,0.9])))}')
                # print(data.min(), data.max(), np.mean(data), np.median(data), np.std(data), np.quantile(data, [0.1,0.2,0.5,0.75,0.9]))
                print(nc1,nc2,nc3,nc4)
                print()
            # quit()
            if COMPRESS:
                tic = time.time()
                dataset.ttICEstar(
                    data,
                    heuristicsToUse=HEURISTICS,
                    occupancyThreshold=1,
                )
                step_time = time.time()-tic
                total_time += step_time
                print(f'{simulation_idx:4d}, {dataset.compressionRatio:09.3f}, {np.prod(dataset.reshapedShape)/dataset.ttCores[-1].shape[0]:12.3f}, {dataset.ttCores[-1].shape[0]:4d}, {step_time:07.3f}, {total_time:07.3f}, {dataset.ttRanks}')
                # print(simulation_idx, dataset.compressionRatio, np.prod(dataset.reshapedShape)/dataset.ttCores[-1].shape[0], dataset.ttCores[-1].shape[0], step_time, total_time, dataset.ttRanks)
            
        if COMPRESS:
            dataset.saveData(
                f"warpx_ni_compressed_{NORMALIZATION_1}_{NORMALIZATION_2}_eps"+f"{EPS:0.8f}".split(".")[-1],
                directory = "../results/warpx/",
                outputType = "ttc",
            )
            latent_data = np.zeros((dataset.ttRanks[-2], warpx_data.shape[-1] - idx_ss))
            for idx,simulation_idx in enumerate(range(idx_ss, warpx_data.shape[-1])):
                data = warpx_data[..., simulation_idx][..., None]
                data, nc1, nc2 = normalize(data, NORMALIZATION_1)
                data, nc3 ,nc4 = normalize(data, NORMALIZATION_2)
                # print(latent_data[... , idx].shape , dataset.projectTensor(data).shape)
                latent_data[... , idx] = dataset.projectTensor(data).squeeze()
            np.save(
                f"../results/warpx/warpx_ni_latent_data_{NORMALIZATION_1}_{NORMALIZATION_2}_eps"+f"{EPS:0.8f}".split(".")[-1]+".npy", latent_data
            )
