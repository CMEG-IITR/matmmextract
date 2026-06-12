Getting Started
===============

Installation
------------

.. code-block:: bash

   pip install -e .

Quick Example
-------------

Fetch OpenAlex papers:

.. code-block:: python

   from paper_pipeline.openalex import fetch_elsevier

   result = fetch_elsevier(
       keywords=["titanium alloy", "microstructure"],
       from_year=2020,
       to_year=2024,
       max_results=500,
   )

   print(result.df.head())

See the Examples section for complete workflows.
