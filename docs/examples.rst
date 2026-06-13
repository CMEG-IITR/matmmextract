Examples
========

Elsevier Full Pipeline
----------------------

.. literalinclude:: ../examples/elsevier_full.py
   :language: python
   :linenos:

Elsevier + Scopus Workflow
--------------------------

.. literalinclude:: ../examples/elsevier_scopus.py
   :language: python
   :linenos:

Springer Full Pipeline
----------------------

.. literalinclude:: ../examples/springer_full.py
   :language: python
   :linenos:

Springer + Scopus Workflow
--------------------------

.. literalinclude:: ../examples/springer_scopus.py
   :language: python
   :linenos:

Cleaner
-------

.. code-block:: python

   from multimat.inference import cleaner

   cleaner.clean(dry_run=True)
   # cleaner.clean()

