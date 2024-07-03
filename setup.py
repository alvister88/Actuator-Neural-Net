from setuptools import setup, find_packages
import codecs
import os.path


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setup(
    name='motor_nn',
    version='0.0.0',
    description='Neural Network for system identification of motors.',
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    author='Alvin Zhu',
    license='MIT',
    project_urls={'GitHub':'https://github.com/alvister88/Motor-Neural-Net.git'},
    packages=find_packages(include=['motor_nn', 'motor_nn.*']),
    install_requires=requirements,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
    ],
)