from setuptools import find_packages, setup


# Required dependencies
required = [
    # Please keep alphabetized
    'gym>=0.15.4',
    'mujoco-py<2.1,>=2.0',
    'numpy>=1.18',
]


setup(
    name='learning_to_be_taught',
    packages=find_packages(),
    include_package_data=True,
    install_requires=required,
)
