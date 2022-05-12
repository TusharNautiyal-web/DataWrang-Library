from setuptools import setup, find_packages

classifiers = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Education',
    'Framework :: Jupyter :: JupyterLab',
    'Operating System :: Microsoft :: Windows :: Windows 10',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3']

setup(
    name = 'DataWrang',
    version= '0.0.3',
    description = 'Python Library For Data Science',
    author = 'Tushar Nautiyal',
    long_description= 'Python Library For Data Science for data cleaning, preproecessing , feature enginnering and feature selection. It includes multiple functions to deal with multiple scenarios. ',
    author_email='tsa.vevo.music@gmail.com',
    license='MIT',
    classifiers=classifiers,
    keywords='Data',
    packages=['datawrang'],
    install_requires = ['numpy','pandas','colorama']
)