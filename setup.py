from setuptools import setup, find_packages

install_requires = [
    'beautifulsoup4',
    'matplotlib',
    'numpy',
    'pandas',
    'requests',
    'tabulate',
    'torch',
    'scikit-learn'
  ]

setup(
    name='timeseries-predict',
    version='0.0.1',
    author='Dunnchadn Strnad, Bob Weigel',
    author_email='dstrnad@gmu.edu',
    packages=find_packages(),
    url='https://github.com/rweigel/timeseries-predict',
    include_package_data=True,
    install_requires=install_requires
)