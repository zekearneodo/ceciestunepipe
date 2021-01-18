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
                        'seaborn',
                        'spikeinterface',
                        'scipy',
                        'tqdm',
                       ],
      zip_safe=False)
