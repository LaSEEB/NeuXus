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

Update the package
##################

To update the Pypi on-line package, run:
::

   $ python setup.py sdist bdist_wheel
   $ twine upload dist/*

Update the doc
##############

Set up the doc-repo:
::

   $ mkdir NeuXus-doc
   $ cd NeuXus-doc
   $ git clone https://github.com/LaSEEB/NeuXus.git html
   $ cd html
   $ git checkout -b gh-pages remotes/origin/gh-pages

Your repos should look like:

| NeuXus
| ├── docs
| ├── examples
| ├── neuxus
| ├── tests
| └── data
| NeuXus-doc
| └── html
| 
| 

To compile the doc, run from the original repo:
::

   $ cd docs
   $ make html

This will update the local documentation.

To update online documentation:
::

   $ git commit -a -m "rebuilt docs"
   $ git push origin gh-pages
