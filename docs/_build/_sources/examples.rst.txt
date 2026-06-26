Examples
========

Elsevier Full Pipeline (OpenAlex → Elsevier → Detection → Azure → Dataset)
---------------------------------------------------------------------------

.. literalinclude:: ../examples/elsevier_full.py
   :language: python
   :linenos:

Elsevier from Scopus Export (Scopus → Elsevier → Detection → Azure → Dataset)
------------------------------------------------------------------------------

.. literalinclude:: ../examples/elsevier_scopus.py
   :language: python
   :linenos:

Springer Full Pipeline (OpenAlex → Springer → Detection → Gemini → Dataset)
----------------------------------------------------------------------------

.. literalinclude:: ../examples/springer_full.py
   :language: python
   :linenos:

Springer from Scopus Export (Scopus → Springer → Detection → Azure → Dataset)
------------------------------------------------------------------------------

.. literalinclude:: ../examples/springer_scopus.py
   :language: python
   :linenos:

Cleanup Intermediate Files
--------------------------

.. code-block:: python

   from matmmextract.inference import clean

   clean(dry_run=True)
   # clean()

