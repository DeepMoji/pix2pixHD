import os
from os import listdir
from os.path import isfile, join
import logging
from time import sleep


if __name__ == "__main__":
    print('Bring training data')

    # Go over the toons file and bring from the cloud the missing raw files
    toons_dir = '/root/code/pix2pixHD/datasets/toonify/train_B' # should be input later
    raw_dir = '/home/pix2pixHD/datasets/toonify/raw' # should be input later
    aligned_dir = '/root/code/pix2pixHD/datasets/toonify/train_A/'  # should be input later

    # logging.basicConfig(level=logging.INFO, filename='/home/pix2pixHD/bring.log')
    # logging.info('start bringing')

    toon_files = [f for f in listdir(toons_dir) if isfile(join(toons_dir, f))]


    files_list = ''
    files_cnt = 0
    for toon in toon_files:
        # Check if the corresponding file exists in aligned folder
        # remove the '-toon' from the filename
        if 'toon' not in toon:
            continue
        aligned_name = toon[0:8] + '.png'
        if isfile(join(aligned_dir, aligned_name)):
            continue
        # copy 200 files
        files_list = files_list + ' gs://deepmoji.appspot.com/mk_results/aligned_ffhq/' + aligned_name
        files_cnt = files_cnt + 1
        if files_cnt >= 1000:
            print('copying ...')
            os.system('gsutil -m cp ' + files_list + ' ' + aligned_dir)
            files_cnt = 0
            files_list = ''
            sleep(5)
        # If not bring it
        # os.system('gsutil cp gs://deepmoji.appspot.com/mk_results/aligned_ffhq/' + aligned_name + ' ' +
        #           join(aligned_dir, aligned_name))
        # print(toon)
        # logging.info('Brought ' + aligned_name)
        pass
    if files_cnt > 0:
        os.system('gsutil -m cp ' + files_list + ' ' + aligned_dir)