import numpy as np


def is_notebook():
    from IPython import get_ipython

    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


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

def train_test_split(ids, n_train, frames_all,info_key='dft_energy',array_key='dft_force'):
    """Perform a train-test split

    Parameters
    ----------

    ids: list(int), size N
        Ordering (e.g. random shuffle) for selecting the split
    n_train: int
        Number of structures in the training set
    frames_all: list(ase.Atoms), length N
        List of all frames in the dataset
    y_all: list(float) or 1-D array, length N
        Set of all function values (energies) in the dataset
    f_all: list or 3-D array, size (N, M, 3)
        Set of all forces (negative gradients) in the dataset
        M is the maximum number of atoms in any one structure

    Returns
    -------

    ((list(Atoms): frames_train, 1-D array: y_train, 3-D array: f_train),
     (list(Atoms): frames_test, 1-D array: y_test, 3-D array: f_test))
    """
    train_ids = ids[:n_train]
    test_ids = ids[n_train:]

    frames_train = [frames_all[ii] for ii in train_ids]
    frames_test = [frames_all[ii] for ii in test_ids]
    y_train, f_train = extract_ref(frames_train,info_key=info_key, array_key=array_key)
    y_test, f_test = extract_ref(frames_test,info_key=info_key, array_key=array_key)

    return frames_train, y_train, f_train, frames_test, y_test, f_test

def fix_frames(frames):
    for frame in frames:
        frame.wrap(eps=1e-11)
        aa = {}
        for k,v in frame.info.items():
            aa[k.lower()] = v
        frame.info = aa
        aa = {}
        for k,v in frame.arrays.items():
            aa[k.lower()] = v
        frame.arrays = aa
    return frames

def filter_frames(frames, exclude):
    filtered_frames = []
    for frame in frames:
        if frame.info['config_type'] not in exclude:
            filtered_frames.append(frame)
    return filtered_frames

def get_config_type(frames):
    config_types = set()
    for frame in frames:
        config_types.add(frame.info['config_type'])
    return config_types

dft_ref = {}
dft_ref['diamond'] = {"bulk_modulus": 88.596696666666602,
                        "c12": 56.25008999999995,
               "c11": 153.28990999999988,
               "E_vs_V": [[17.813109023568057, -163.06663280425], [18.322054995670012, -163.110046112], [18.831000967771949, -163.1415028745], [19.339946939873919, -163.162464704125], [19.848892911975845, -163.174242557], [20.3578388840778, -163.177966801875], [20.866784856179741, -163.17463056225], [21.375730828281693, -163.165132595375], [21.884676800383637, -163.150242437875], [22.393622772485578, -163.130671150875], [22.902568744587512, -163.107009425875], [23.411514716689464, -163.079814823625]],
               "c44": 72.176929999999999, "a0": 5.4610215037046075}
dft_ref['beta-Sn'] = {"E_vs_V": [[13.462620304548336, -162.75664626205], [13.847236220240584, -162.7954601585], [14.231905420366559, -162.8239554888], [14.616528955757214, -162.84260051285], [15.00117439209707, -162.85309453065], [15.385871808743351, -162.85631297195], [15.770478826496822, -162.85333965315], [16.155054108065379, -162.84486224], [16.539436349955054, -162.8314338523], [16.924316511337032, -162.8142848878], [17.30841204863296, -162.79382098635], [17.692092233780102, -162.7710455315]]}