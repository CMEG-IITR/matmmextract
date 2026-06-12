Examples
========

Elsevier Pipeline
-----------------

.. literalinclude:: ../examples/elsevier_full.py
   :language: python
   :linenos:

Springer Pipeline
-----------------

.. literalinclude:: ../examples/springer_full.py
   :language: python
   :linenos:

Cleaner
-------

.. code-block:: python

   from multimat.inference import cleaner

   cleaner.clean(dry_run=True)
   # cleaner.clean()
