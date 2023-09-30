

import pandas as pd

data = pd.read_csv('f:/imdb_sample/texts.csv')

print(data)

print(data.shape)
texts = list(set(data['text']))

file_name = 'testing.txt'
with open(file_name, 'w', encoding='utf-8') as f:
    f.write(" |EndOfText|\n".join(texts))


weights_dir = "output"

cmd = '''
python transformers/examples/tensorflow/language-modeling/run_clm.py \
    --model_name_or_path distilgpt2 \
    --train_file {0} \
    --do_train \
    --num_train_epochs 3 \
    --overwrite_output_dir \
    --per_device_train_batch_size 2 \
    --output_dir {1}
'''.format(file_name, weights_dir)

import subprocess

# 示例：执行 'dir' 命令（Windows）或 'ls' 命令（macOS/Linux）
command = cmd

print (cmd)

# 执行命令
result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# 输出执行结果
print("执行状态码:", result.returncode)
print("命令输出:")
print(result.stdout)
print("错误输出:")
print(result.stderr)

