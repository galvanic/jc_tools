from setuptools import setup

setup(
    name='JustineTools',
    version='0.1dev',
    description='Helper classes and functions to import into other projects',
    author='Justine Courty',
    author_email='justinecourty@gmail.com',
    url='ssh://git@github.com/galvanic/jc_tools',
    packages=[
        'jc_tools',
        ],
    long_description=open('README.md').read(),
    install_requires=[
        'numpy>=1.14.1',
        ],
    )

