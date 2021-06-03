import os
import sys
import glob

if __name__ == '__main__':
 
    print('running score prediction...')
    train=[

        'evidence.txt_proc_2_train.pkl'

    ]

    test=[

        'evidence.txt_proc_2_test.pkl'

    ]

    for ttrain, ttest in zip(train,test):
        mdir = '_'.join(ttrain.split('_')[:-1])
        mdir = 'out/' + '_'.join(mdir.split('/')[1:]) + '/'
        # mdir = 'out/long' + '_'.join(mdir.split('/')[1:]) + '/'
        os.system('python bidirectional_lstm.py {} {} {}'.format(mdir, ttrain, ttest))
