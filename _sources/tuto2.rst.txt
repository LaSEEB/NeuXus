
.. _tutorial2:

Create your own node
====================

NeuXus basis
------------

Before writing your own Node, you need to understand a bit how the softare works.

.. image:: image/software.png
  :width: 700
  :alt: Alternative text

NeuXus first loads the pipeline script. It then repeates infinetely a loop that clears all Ports and updates each Node. The ``update()`` method of each Node is called at each iteration and get data from the input Port. NeuXus is stopped when ``Esc`` is pressed. All terminate function are then called.

A Node has then 3 methods: ``__init__()``, ``update()`` and ``terminate()``:

.. image:: image/node.png
  :width: 700
  :alt: Alternative text

A Node can have a Port as input and/or output. Ports can share different kind of Data stored in Pandas DataFrame (table). A Port is an iteration object, it means that to get its values, you need to iter over it:

::

   port = Port()
   for chunk in port:
      ...

Type of ports are:

   * ``'signal'``: 1 iteration = 1 chunk of the continuous signal
   * ``'epoch'``: 1 iteration = 1 epoch
   * ``'marker'``: 1 iteration = 1 marker
   * ``'spectrum'``: 1 iteration = 1 spectrum computed at one timestamp

A Node template
---------------

Get inspiration from the following :download:`template <../pipeline_template.py>` to write new pipelines.

.. literalinclude:: ../node_template.py
   :language: python

.. _Github: https://github.com/LaSEEB/NeuXus

Feel free to share your Nodes to our community on Github_