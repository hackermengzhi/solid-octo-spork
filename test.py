import os
import subprocess
from tqdm import tqdm

# dataset_path
train_path = './ST-100'
test_path='./ST-100'
# wav_files for training and testing
test_files = [f for f in os.listdir(test_path) if f.endswith('.wav')]
#subprocess.run(['python3', 'demo.py', '--file', train_path, '--mode', 'register'])

# Recognize the testing files
correct = 0
for file_name in tqdm(test_files, desc='Recognizing'):
    file_path = os.path.join(test_path, file_name)
    result = subprocess.run(['python3', 'main.py', '--file', file_path, '--mode', 'register'], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    recognized_name = output.strip().split('\n')[-1]

