from setuptools import setup

setup(name='ceciestunepipe',
      version='0.1',
      description='Spike sorting pipeline based on spikeinterface for spikeglx and openephys',
      url='http://github.com/zekearneodo/ceciestunepipe',
      author='Zeke Arneodo',
      author_email='ezequiel@ini.ethz.ch',
      license='MIT',
      packages=['ceciestunepipe'],
      install_requires=['numpy',
                        'matplotlib',
                        'pandas>=0.23',
                        'more_itertools',
                        'peakutils>=1.3',
                        'librosa',
                        'seaborn',
                        'scipy',
                        'tqdm',
                        'h5py',
                        'parse'
                       ],
      zip_safe=False)
