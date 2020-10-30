import os
from os import listdir
from os.path import isfile, join
import logging


if __name__ == "__main__":
    print('Bring training data')

    # Go over the toons file and bring from the cloud the missing raw files
    toons_dir = '/home/pix2pixHD/datasets/toonify/toons' # should be input later
    raw_dir = '/home/pix2pixHD/datasets/toonify/raw' # should be input later
    aligned_dir = '/home/pix2pixHD/datasets/toonify/aligned'  # should be input later

    logging.basicConfig(level=logging.INFO, filename='/home/pix2pixHD/bring.log')
    logging.info('start bringing')

    toon_files = [f for f in listdir(toons_dir) if isfile(join(toons_dir, f))]
    for toon in toon_files:
        # Check if the corresponding file exists in aligned folder
        # remove the '-toon' from the filename
        if 'toon' not in toon:
            continue
        aligned_name = toon[0:8] + '.png'
        if isfile(join(aligned_dir, aligned_name)):
            continue
        # If not bring it
        os.system('gsutil cp gs://deepmoji.appspot.com/mk_results/aligned_ffhq/' + aligned_name + ' ' +
                  join(aligned_dir, aligned_name))
        logging.info('Brought ' + aligned_name)
        pass