
import glob
import os
from utilities.print_utils import *

def get_mappings(root_dir, files, annot_name):
    pairs = []
    for f in files:
        f = f.replace(root_dir, '/')
        img_f = f.replace(annot_name, 'leftImg8bit')
        img_f = img_f.replace('_labelTrainIds.png', '.png')
        if not os.path.isfile(root_dir + img_f):
            print_error_message('{} file does not exist. Please check'.format(root_dir + img_f))
            exit()
        line = img_f + ','  + f
        pairs.append(line)
    return pairs

def main(cityscapesPath, split):
    searchFine = os.path.join(cityscapesPath, "gtFine", split, "*", '*_labelTrainIds.png')
    filesFine = glob.glob(searchFine)
    filesFine.sort()

    if not filesFine:
        print_warning_message("Did not find any files. Please check root directory: {}.".format(cityscapesPath))
        fine_pairs = []
    else:
        print_info_message('{} files found for {} split'.format(len(filesFine), split))
        fine_pairs = get_mappings(cityscapesPath, filesFine, 'gtFine')

    if not fine_pairs:
        print_error_message('No pair exist. Exiting')
        exit()
    else:
        print_info_message('Creating train and val files.')
    f_name = split + '.txt'
    with open(os.path.join(cityscapesPath, f_name), 'w') as txtFile:
        for pair in fine_pairs:
            txtFile.write(pair + '\n')
    print_info_message('{} created in {} with {} pairs'.format(f_name, cityscapesPath, len(fine_pairs)))

    if split == 'train':
        split_orig = split
        split = split + '_extra'
        searchCoarse = os.path.join(cityscapesPath, "gtCoarse", split, "*", '*_labelTrainIds.png')
        filesCoarse = glob.glob(searchCoarse)
        filesCoarse.sort()
        if not filesCoarse:
            print_warning_message("Did not find any files. Please check root directory: {}.".format(cityscapesPath))
            course_pairs = []
        else:
            print_info_message('{} files found for {} split'.format(len(filesCoarse), split))
            course_pairs = get_mappings(cityscapesPath, filesCoarse, 'gtCoarse')
        if not course_pairs:
            print_warning_message('No pair exist for coarse data')
            return
        else:
            print_info_message('Creating train and val files.')
        f_name = split_orig + '_coarse.txt'
        with open(os.path.join(cityscapesPath, f_name), 'w') as txtFile:
            for pair in course_pairs:
                txtFile.write(pair + '\n')
        print_info_message('{} created in {} with {} pairs'.format(f_name, cityscapesPath, len(course_pairs)))

if __name__ == '__main__':
    cityscapes_path = '../../../vision_datasets/cityscapes/'
    main(cityscapes_path, "train")
    main(cityscapes_path, "val")