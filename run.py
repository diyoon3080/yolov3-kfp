import glob
import os

scripts = glob.glob('./component-files-python/*.py')

for script in scripts:
    os.system('python3 ' + script)

os.system('python3 create-pipeline.py')
