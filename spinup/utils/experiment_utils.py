import os.path as osp
import pathlib

import joblib
import torch

from spinup.utils.general_utils import get_latest_file_iteration


def load_env_and_agent(experiment_folder, iteration='last', device=torch.device('cpu')):
    """
    Load a policy from save, whether it's TF or PyTorch, along with RL env.

    Not exceptionally future-proof, but it will suffice for basic uses of the
    Spinning Up implementations.

    Checks to see if there's a tf1_save folder. If yes, assumes the model
    is tensorflow and loads it that way. Otherwise, loads as if there's a
    PyTorch save.
    """

    # handle which epoch to load from
    if iteration == 'last':
        _, i = get_latest_file_iteration(osp.join(experiment_folder, 'pyt_save'),
                                         pattern='agent*.pt')
        if i:
            itr = '%d' % i
        else:
            itr = ''
    else:
        assert isinstance(iteration, int), \
            "Bad value provided for itr (needs to be int or 'last')."
        itr = '%d' % iteration

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(experiment_folder, 'vars' + itr + '.pkl'))
        env = state['env']
    except:
        env = None

    agent = load_pytorch_obj(experiment_folder, itr, 'agent', device=device)
    return env, agent


def get_latest_saved_file(save_dir, prefix):
    save_dir = pathlib.Path(save_dir)
    fp, _ = get_latest_file_iteration(save_dir, prefix + '*.pt')
    if fp:
        return fp

    fp = save_dir / (prefix + '.pt')
    if fp.exists():
        return fp

    return None


def load_pytorch_obj(experiment_folder, itr, obj_name, device=torch.device('cpu')):
    filepath = osp.join(experiment_folder, 'pyt_save', obj_name + itr + '.pt')
    print('\n\nLoading from %s.\n\n' % filepath)
    return torch.load(filepath, map_location=device)
