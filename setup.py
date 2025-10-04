from setuptools import setup, find_packages

setup(
    name='ml_autopipeline',
    version='0.1.0',
    description='Automated ML pipeline with EDA, bias detection, sampling, and model training for beginners',
    author='Divyansh Saxena',
    packages=find_packages(),
    install_requires=[
        'pandas>=1.0',
        'scikit-learn>=0.24',
        'imbalanced-learn>=0.8',
    ],
    entry_points={
        'console_scripts': [
            'ml-autopipeline=ml_autopipeline.cli:main',
        ],
    },
    python_requires='>=3.7',
)
