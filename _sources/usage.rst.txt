Usage
=====

Basic usage
-----------

Run basics examples from NeuXus, use ``--example`` command line option:
::

   $ neuxus -e basics/generate_send.py

| examples
| ├── basics
| │   ├── generate_send.py
| │   ├── receive_graz.py
| │   └── stimulate_send.py
| ├── motor_imagery_simple
| │   ├── 0_get-raw.py
| │   ├── 1_data_acquisition.py
| │   ├── 2_train_lda.py
| │   └── 3_online-mi.py
| └── test
|     └── sdsd.py
| 
| 

Press ``Esc`` to quit the software and stop the running.

Run your own pipeline
---------------------

Refer to to learn how to create a pipeline
Launch your pipeline with:
::

   $ neuxus my_pipeline.py

Command line options
--------------------

Change the log level with (default level is INFO):
::

   $ neuxus my_pipeline.py -l DEBUG

This will display what data enters each Node

To save logs in a file, run:
::

   $ neuxus my_pipeline.py -l DEBUG -f

The file is saved in the same directory you run NeuXus