# NeuXus

NeuXus is a modular software written in Python that allows to easily create and execute pipelines in real time. NeuXus is specifically designed for BCI real-time processing and classification.

It offers a lot of possibilities, many basic functions are already implemented so that the first pipelines can be quickly created. For more specific functions, users can develop and use their own functions.


## Installation

### Basic use

NeuXus requires Python 3.7+, verify your Python version with:
```
python --version
```
Install NeuXus with:
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

For more information, read the [Documentation](https://laseeb.github.io/NeuXus/index.html)
