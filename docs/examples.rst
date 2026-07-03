.. meta::
   :description lang=en:
      Complete examples demonstrating OpenAlex, Elsevier, Springer,
      Scopus, panel detection, caption generation, and dataset construction.

   :keywords:
      examples,
      tutorial,
      materials science,
      multimodal dataset

Examples
========

Elsevier Full Pipeline (OpenAlex → Elsevier → Detection → Azure → Dataset)
---------------------------------------------------------------------------------------------------------------

.. literalinclude:: ../examples/elsevier_full.py
   :language: python
   :linenos:

Elsevier from Scopus Export (Scopus → Elsevier)
-----------------------------------------------

.. literalinclude:: ../examples/elsevier_scopus.py
   :language: python
   :linenos:

Springer Full Pipeline (OpenAlex → Springer → Detection (model checkpoint from Hugging Face Hub) → Gemini → Dataset)
--------------------------------------------------------------------------------------------------------------------

.. literalinclude:: ../examples/springer_full.py
   :language: python
   :linenos:

Springer from Scopus Export (Scopus → Springer)
-----------------------------------------------

.. literalinclude:: ../examples/springer_scopus.py
   :language: python
   :linenos:

Cleanup Intermediate Files
--------------------------

.. code-block:: python

   from matmmextract.inference import clean

   clean(dry_run=True)
   # clean()

