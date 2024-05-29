from collections import defaultdict
import glob
import os
import numpy as np
import random
import json

def make_noisy_label_dict(data_name, data_path, data_type, noise_ratio, dict_save_path, seed=42):
    np.random.seed(seed)
    random.seed(seed)

    data_path_list = sorted(glob.glob(os.path.join(data_path, '*', '*')))

    class_list = set([x.split('/')[-2] for x in data_path_list])
    class_to_truelabel = {c: float(i) for i,c in enumerate(sorted(list(class_list)))}
    cls_num = len(class_list)

    # Generate noisy label index
    noise_label_idx = np.random.choice( range(len(data_path_list)), int((len(data_path_list))*noise_ratio), replace=False)

    data_dict=defaultdict(list)
    for i, img_path in enumerate(data_path_list):
        class_name = img_path.split('/')[-2]
        true_label = class_to_truelabel[class_name]

        if i in noise_label_idx:
            label_list = np.arange(cls_num)

            # Delete the true label within whole label list
            label_list = np.delete(label_list, int(true_label))
            noisy_label = float(random.choice(label_list))
            data_dict['noisy_label'].append(noisy_label)
        else:
            data_dict['noisy_label'].append(float(true_label))

        data_dict['true_label'].append(float(true_label))
        data_dict['image_path'].append(img_path)

    save_dict_name = f'{data_name}_{data_type}_NoiseRatio_{noise_ratio}.json'
    dict_save_path = os.path.join(dict_save_path, save_dict_name)
    with open(dict_save_path, 'w', encoding='utf-8') as json_file:
        json.dump(data_dict, json_file, indent='\t')