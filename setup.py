import os
from setuptools import setup, Command, find_packages


class CleanCommand(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')


cmdclass = {'clean': CleanCommand}


options = {
    'name': 'phenotype_cover',
    'description': 'phenotype_cover is a package for biomarker discovery using multiset multicover.',
    'long_description': 'Implements the greedy and cross-entropy-method phenotype cover algorithms.',
    'license': 'MIT',
    'version': '0.2',
    'author': 'Euxhen Hasanaj',
    'author_email': 'ehasanaj@cs.cmu.edu',
    'url': 'https://github.com/euxhenh/phenotype-cover',
    'provides': ['phenotype_cover'],
    'package_dir': {'phenotype_cover': 'src/phenotype_cover'},
    'packages': find_packages(where='src'),
    'cmdclass': cmdclass,
    'platforms': 'ALL',
    'keywords': ['biomarker', 'marker', 'phenotype', 'scRNA-seq', 'set', 'cover', 'multiset', 'multicover'],
    'install_requires': ['numpy', 'matplotlib', 'scikit-learn', 'multiset-multicover'],
    'python_requires': ">=3.7"
}

setup(**options)
