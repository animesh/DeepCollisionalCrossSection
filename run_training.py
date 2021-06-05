import os
import sys
import glob
import wandb
wandb.init(project="deepScore")
#wandb.log({'Loss': loss})
device_name = "gpu"  # Choose device from cmd line. Options: gpu or cpu
if device_name == "gpu":
    device_name = "/gpu:0"

if __name__ == '__main__':
 
    print('running ccs')
    train=[

        'evidence.txt_proc_2_test.pkl'

    ]

    test=[

        'evidence.txt_proc_2_test.pkl'

    ]

    for ttrain, ttest in zip(train,test):
        mdir = '_'.join(ttrain.split('_')[:-1])
        mdir = 'out/' + '_'.join(mdir.split('/')[1:]) + '/'
        # mdir = 'out/long' + '_'.join(mdir.split('/')[1:]) + '/'
        os.system('python bidirectional_lstm.py {} {} {}'.format(mdir, ttrain, ttest))
