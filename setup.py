import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as fr:
    requirements = fr.read().split('\n')

setuptools.setup(
    name="neuxus",
    version="0.0.3",
    author="S.Legeay, A.Vourvopoulos",
    author_email="legeay.simon.sup@gmail.com",
    description="A flexible software to build real-time pipeline for EEG processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LaSEEB/NeuXus",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    packages=setuptools.find_packages(exclude=['contrib', 'docs', 'test*']),
    install_requires=requirements,
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'neuxus=neuxus.main:main'],
    }
)
