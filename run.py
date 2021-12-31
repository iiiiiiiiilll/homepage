# coding: utf-8
import os
import shutil

if os.path.isdir('build'):
    shutil.rmtree('build')
    print(1)
os.system('make html')


