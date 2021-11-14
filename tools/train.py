import numpy as np
from ase.io import read

from rascal.representations import SphericalInvariants
from rascal.models import gaptools
from rascal.utils import dump_obj, load_obj, get_score
from rascal.models import gaptools, KRR

from .download import download_url
from .utils import fix_frames, train_test_split, filter_frames, get_config_type

def extract_ref(frames,info_key='dft_formation_energy_per_atom_in_eV',array_key='zeros'):
    y,f = [], []
    for frame in frames:
        y.append(frame.info[info_key])
        if array_key is None:
            pass
        elif array_key == 'zeros':
            f.append(np.zeros(frame.get_positions().shape))
        else:
            f.append(frame.get_array(array_key))
    y= np.array(y)
    try:
        f = np.concatenate(f)
    except:
        pass
    return y,f

def sparcify(soap, frames, n_features, n_sparse):
    hypers = soap._get_init_params()
    soap, feature_list = gaptools.calculate_representation(frames, hypers)

    sparcified_hypers = gaptools.sparsify_features(soap, feature_list, n_features, selection_type="FPS")

    soap, feature_list = gaptools.calculate_representation(frames, sparcified_hypers)

    X_sparse = gaptools.sparsify_environments(soap, feature_list, n_sparse, selection_type="FPS")
    return soap, X_sparse, feature_list

def train(frames, soap, n_features=200, n_sparse={14:1000}, lambdas=[1e-3,1e-2]):

    isolated_atom = frames[0]
    frames = frames[1:]

    global_species = [14]

    energy_baseline = {
        14: isolated_atom.info['dft_energy'],
    }



    soap, X_sparse, feature_list = sparcify(soap, frames, n_features, n_sparse)

    (k_obj, K_sparse_sparse, K_full_sparse, K_grad_full_sparse) = gaptools.compute_kernels(
        soap,
        feature_list,
        X_sparse,
        soap_power=2,
    )

    natoms = [len(frame) for frame in frames]
    y, f = extract_ref(frames,info_key='dft_energy', array_key='dft_force')


    weights = gaptools.fit_gap_simple(
        [frames[ii] for ii in train],
        K_sparse_sparse,
        y[train],
        K_full_sparse,
        energy_regularizer_peratom=lambdas[0],
        forces=f[train],
        kernel_gradients_sparse=K_grad_full_sparse,
        energy_atom_contributions=energy_baseline,
        force_regularizer=lambdas[1],
        solver="RKHS-QR"
    )

    model = KRR(weights, k_obj, X_sparse, energy_baseline,
            description="GAP MLIP for Silicon")

    return model

def score(cv, frames, exclude_configs, soap, n_features=200, n_sparse={14:1000}, lambdas=[[1e-3,1e-2]]):

    isolated_atom = frames[0]
    frames = frames[1:]

    global_species = [14]

    energy_baseline = {
        14: isolated_atom.info['dft_energy'],
    }

    frames = filter_frames(frames, exclude_configs)


    soap, X_sparse, feature_list = sparcify(soap, frames, n_features, n_sparse)

    (k_obj, K_sparse_sparse, K_full_sparse, K_grad_full_sparse) = gaptools.compute_kernels(
        soap,
        feature_list,
        X_sparse,
        soap_power=2,
    )

    natoms = [len(frame) for frame in frames]
    y, f = extract_ref(frames,info_key='dft_energy', array_key='dft_force')

    models = []
    scores = []
    for train, test in cv.split(y.reshape((-1,1))):

        K_train_sparse, K_grad_train_sparse = gaptools.extract_kernel_indices(
            train, K_full_sparse, K_grad_full_sparse, natoms=natoms
        )
        K_test_sparse, K_grad_test_sparse = gaptools.extract_kernel_indices(
            test, K_full_sparse, K_grad_full_sparse, natoms=natoms
        )

        for (lambda_E, lambda_F) in lambdas:
            weights = gaptools.fit_gap_simple(
                [frames[ii] for ii in train],
                K_sparse_sparse,
                y[train],
                K_train_sparse,
                energy_regularizer_peratom=lambda_E,
                forces=f[train],
                kernel_gradients_sparse=K_grad_train_sparse,
                energy_atom_contributions=energy_baseline,
                force_regularizer=lambda_F,
                solver="RKHS-QR"
            )

            model = KRR(weights, k_obj, X_sparse, energy_baseline,
                    description="")

            models.append(model)
            frames_test = [frames[ii] for ii in test]
            y_pred = model.predict(frames_test, KNM=K_test_sparse)
            f_pred = model.predict_forces(frames_test, KNM=K_grad_test_sparse)
            natoms = np.array([len(frame) for frame in frames_test])
            baseline = energy_baseline[14]
            sc = get_score(y_pred/natoms-baseline, y[test]/natoms-baseline)
            sc = {k+'_E' : v for k,v in sc.items()}
            sc['E_unit'] = 'eV'

            sc_f = get_score(f_pred.flatten(), f[test].flatten())
            sc.update(**{k+'_F' : v for k,v in sc_f.items()})
            sc['F_unit'] = 'eV/A'
            sc.update(**{'lambda_E': lambda_E, 'lambda_F': lambda_F})
            sc.update(n_features=n_features, n_sparse=tuple((k,v)for k,v in n_sparse.items()) )
            scores.append(sc)

    return scores

    