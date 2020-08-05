.. _contribute:

Contribute
==========

For developers
--------------
To develop new features for NeuXus, make sure you have Python3.7+.

Clone from source:
::

   $ git clone https://github.com/LaSEEB/NeuXus.git
   $ pip install -r requirements.txt
   $ pip install -r dev_requirements.txt

Create the tar.gz file and install it on your computer:
::

   $ setup.py sdist
   $ cd dist
   $ pip install nexus-xx.xx.xx.tar.gz

When creating new Nodes, write tests in order to be sure that you don't modify input data. Take example on already existant tests.

Launch tests with:
::

   $ python -m unittest discover -v

For administrators
------------------

To update the Pypi on-line package, run:
::

   $ python setup.py sdist bdist_wheel
   $ twine upload dist/*

To compile the doc:
::

   $ cd docs
   $ make html

To update online documentation:
