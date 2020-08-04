
.. _tutorial1:

Create a pipeline
=================

A simple example
----------------

NeuXus loads and executes a py file (called the pipeline script) at the beginning of its execution, this file describes the pipeline. It initializes all Nodes and links between them.

Example:

.. literalinclude:: ../../examples/basics/stimulate_send.py
   :language: python

Try the following command to run this pipeline
::

   $ neuxus -e basics/stimulate_send.py

This pipeline reads the stimulations config with the Stimulator Node first (refer to :ref:`stimulator`), and then send the stimulations via LSL with the second Node LslSend (refer to :ref:`io`). To link the Stimulator to the LslSend, LslSend takes as ``input_port``, the ``output`` attribute of the Stimulator, ``generated_markers.output``.

.. note::
   The ``output`` attribute is of type Port, and enables to share data between Nodes. Not every Node has that attribute, read the :ref:`api`!

.. warning::
   Do not name Nodes with the same name, else they will override each others.

A more complicated example
--------------------------

This pipeline sends by LSL a simple DSP feedback calculated in real-time:

.. literalinclude:: ../../examples/basics/simple_DSP_feedback.py
   :language: python

It first receives the signal from LSL, then filters it with a bandpass :ref:`ButterFilter`, squares it with :ref:`ApplyFunction`. The signal is epoched every 0.5s on a range of 1s and averaged epoch by epoch. The DSP is then sent via LSL.

.. note::
	The pipeline script is only executed once when running NeuXus, users can then define functions, constants that can be used to build Nodes.

.. warning::
	The processing frequency of NeuXus is self-regulated, the software calculates all updates as quickly as possible and then waits for the next data entry.

	* If the RAM is overloaded (because of other running software or because there are too many nodes in the pipeline), the loop takes longer to complete. This may involve a significant (and exponentially increasing) delay.

	* If stimuli are used outside the software (via LSL for example), divide your pipeline, one with the independent stimulator, the other with the rest: this ensures that the pipeline does not slow down the sending of stimulation via LSL.

Write your pipeline
-------------------

Get inspiration from the following :download:`template <../pipeline_template.py>` to write new pipelines.

.. literalinclude:: ../pipeline_template.py
   :language: python

.. note::
	Users can load their own Nodes in pipeline (see :ref:`tutorial2`).