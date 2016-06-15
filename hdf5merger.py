import h5py

target_filename = "target.hdf5"
source_filenames = ["grafy_data.hdf5","grafy_closed.hdf5" ]
with h5py.File(target_filename) as t:
    for filename in source_filenames:
        with h5py.File(filename) as s:
            for dataset_name in s:
                if dataset_name not in t:
                    t.copy(s[dataset_name], dataset_name)
