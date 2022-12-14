1. On windows 
A. If you don't use a python virtual environment, create a batch file with:

@ECHO OFF
neuxus example-python-file.py
PAUSE

B. If you use a python virtual environment, create a batch file with:

@ECHO OFF
"path-to-the-example-env\example-env\Scripts\neuxus.exe" example-python-file.py
PAUSE

