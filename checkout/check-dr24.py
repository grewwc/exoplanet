import pandas as pd
from config import *
import os
from dr25.gen_flux_txt import test_kepid
from clean_utils.normalization import norm_kepid
from models.utils import load_model


def check():
    m = load_model()
    fname = os.path.join(csv_folder, csv_name)
    df = pd.read_csv(fname, comment='#')
    kepids = set(df['kepid'])
    count, total = 1, len(kepids)
    diff_count = 0
    processed = 0
    with open('diff_kepid.txt', 'w') as f:
        with open('unk_kepid.txt', 'w') as f_unk:
            for kepid in kepids:
                res = test_kepid(m, kepid, dr24=1)
                sub_df = df[df['kepid'] == int(kepid)]
                for plnt, prob in res.items():
                    cls = sub_df[sub_df['tce_plnt_num'] == int(
                        plnt)]['av_training_set'].values[0]
                    if cls == 'PC' and prob > 0.5 \
                            or cls != 'PC' and cls != 'UNK' and prob < 0.5:
                        # right += 1
                        processed += 1
                    elif cls != 'UNK':
                        processed += 1
                        diff_count += 1
                        print(f'diff rate: {diff_count/processed:.3f}')
                        print(f'{kepid}-{plnt} prob: {prob}',
                              file=f)
                    else:
                        print(f'{kepid}-{plnt} prob: {prob}',
                              file=f_unk)

                print(f'{count}/{total}')
                count += 1


if __name__ == '__main__':
    check()
