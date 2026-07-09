.. meta::
   :description lang=en:
      Learn how to install MatMMExtract and get started with building
      multimodal materials science datasets from scientific literature
      using OpenAlex, Elsevier, Springer, Scopus, computer vision, and
      large language models.

   :keywords:
      MatMMExtract,
      installation,
      getting started,
      Python,
      materials science,
      multimodal datasets,
      OpenAlex,
      Elsevier,
      Springer,
      Scopus,
      scientific literature,
      figure extraction

Getting Started
===============

This guide walks through installing **MatMMExtract** and preparing your
environment for extracting figures, captions, and multimodal datasets
from scientific literature.

Installation
------------

Install the latest release from PyPI:

.. code-block:: bash

   pip install matmmextract

Or install the latest development version from source:

.. code-block:: bash

   git clone https://github.com/CMEG-IITR/matmmextract.git
   cd matmmextract
   pip install -e .

Verify the installation:

.. code-block:: python

   import matmmextract

Build Documentation
-------------------

To build the documentation locally, run:

.. code-block:: bash

   sphinx-build -b html docs docs/_build

The generated HTML documentation will be available at:

.. code-block:: text

   docs/_build/index.html

Next Steps
----------

After installation, continue with the following guides:

* :doc:`examples` — Complete end-to-end pipeline examples.
* :doc:`api` — Full API reference for all modules.