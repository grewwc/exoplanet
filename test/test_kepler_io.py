import sys
import pytest

from clean_utils.normalization import get_global_fname_by_kepid

import config
# just for add root_dir to PYTHONPATH

from preprocess.kepler_io import *


def load_csv():
    csv_clean_path = os.path.join(csv_folder, csv_name_drop_unk)
    return pd.read_csv(csv_clean_path, comment='#')


@pytest.mark.success
def test_get_PC_IDs():
    all_pcs = get_PC_IDs(np.inf)
    data = load_csv()
    for pc in all_pcs:
        values = data[data['kepid'] == int(pc)]['av_training_set'].values
        has_pc_label = False
        for value in values:
            if value == 'PC':
                has_pc_label = True
                break
        assert has_pc_label, pc

    all_unique_pcs = set(data[data['av_training_set'] == 'PC']['kepid'].values)
    assert len(all_pcs) == len(all_unique_pcs)


@pytest.mark.success
def test_classified_right_in_writing():
    df = pd.read_csv(os.path.join(csv_folder, csv_name_drop_unk), comment='#')
    for choice in range(len(df)):
        row = df.iloc[choice]
        kepid = row['kepid']
        plnt_num = row['tce_plnt_num']
        label = row['av_training_set']
        label = 0 if label == 'PC' else 1
        # reverse the label to test
        abs_name = get_global_fname_by_kepid(kepid, plnt_num, label)
        assert not os.path.exists(abs_name)
