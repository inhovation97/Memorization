import subprocess
import os

# epoch = str(10)
run_file = "./main.py"

#--save_name
#--ttrank

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

cmd_ttlstm = [
    ["python", run_file, '--default_setting', './default_configs.yaml',
                        '--modelname', 'resnet18', 
                        '--dataname', 'cifar10', 
                        # '--wandb',
                        '--noise_ratio', '0.5', 
                        ],
]


for i, cmd in enumerate(cmd_ttlstm):
    p = subprocess.Popen(cmd)
    p.wait()
print('Training is done!')
