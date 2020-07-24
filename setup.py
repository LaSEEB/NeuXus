import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as fr:
    requirements = fr.read().split('\n')

setuptools.setup(
    name="neuxus",
    version="0.0.1",
    author="S.Legeay, A.Vourvopoulos",
    author_email="to do",
    description="A flexible software to build real-time pipeline for EEG processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="to do",
    classifiers=[
        "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    packages=setuptools.find_packages(exclude=['contrib', 'docs', 'test*']),
    install_requires=requirements,
    data_files=[],
    entry_points={
        'console_scripts': [
            'neuxus=neuxus.main:main'],
    }
)
