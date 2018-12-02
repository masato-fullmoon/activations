#bin/bash

inputs_path=${1}
outputs_path="${inputs_path/}/gpu_info.txt"
if [ !-e ${outputs_path} ]; then
    nvidia-smi -i 0 -q > ${outputs_path}
