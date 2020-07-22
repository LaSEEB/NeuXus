# NeuXus
Build flexible pipeline for real-time processing


## Installation

### Basic use

To Do uploading on PyPi
```
pip install neuxus
```

## Run

### Run basic pipeline from NeuXus

```
neuxus basics/generate_send.py -e
```

### Run your pipeline from NeuXus

```
neuxus path_to_my_pipeline.py
```

## Customize your own nodes

see template
TODO

## For developers

Clone from source:

```
git clone https://github.com/athanoid/laseeb-bci.git
```

Install dev-requirements
ToDo

Create the tar.gz file and install it on your computer:
```
setup.py sdist
pip install nexus-xx.xx.xx.tar.gz
```

### Tests

Launch tests with:
```
python -m unittest discover
```
