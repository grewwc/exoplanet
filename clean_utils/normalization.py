from config import train_root_dir
import os


def norm_kepid(kepid):
    return f'{int(kepid):09d}'


def get_global_fname_by_kepid(kepid, plnt_num, label):
    kepid = norm_kepid(kepid)
    label = str(label)
    fname = f'flux_{plnt_num}.txt'
    abs_name = os.path.join(
        train_root_dir, 'flux', label,
        kepid[:4], kepid, fname
    )
    return abs_name
