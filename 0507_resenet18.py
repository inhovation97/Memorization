import subprocess
import os

# epoch = str(10)
run_file = "./train.py"

#--save_name
#--ttrank

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

cmd_ttlstm = [
    ["python", run_file, '--seed', '42', '--model', 'resnet18', '--batch_size', '64', '--lr', '0.01', '--DA', 'none', '--gpu', '--epochs', '100',
     '--scheduler', '--tracking_grads','--data_path', '/media/NAS/DATASET/cifar10/original',
      '--DA', 'flip_crop', '--grad_sample_num', '1024', '--noisy_comb_len', '100',
      '--grad_dict_save_path', '/media/NAS/USERS/inho/Memorization/Exp_correlation', '--model_save_path', '/media/NAS/USERS/inho/Memorization/Exp_correlation'],
]


for i, cmd in enumerate(cmd_ttlstm):
    p = subprocess.Popen(cmd)
    p.wait()
print('Training is done!')
