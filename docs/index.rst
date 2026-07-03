.. meta::
   :description lang=en:
      MatMMExtract is an open-source Python library for building multimodal
      materials science datasets from scientific literature using OpenAlex,
      Elsevier, Springer, figure extraction, panel detection, and LLM captioning.

   :keywords:
      materials science,
      multimodal datasets,
      scientific figure extraction,
      OpenAlex,
      Elsevier,
      Springer,
      Gemini,
      Azure OpenAI,
      computer vision,
      machine learning,
      scientific literature

.. image:: ../logo.svg
   :align: center
   :width: 220px

MatMMExtract
============

**MatMMExtract** is an open-source Python library for building multimodal
materials science datasets from scientific literature.

It provides an end-to-end pipeline for retrieving papers from OpenAlex,
Elsevier, Springer, and Scopus, extracting figures and captions, detecting
scientific figure panels, generating fine-grained captions using modern
large language models (LLMs), and constructing machine-learning-ready
multimodal datasets.

Key Features
------------

* OpenAlex and Scopus paper retrieval
* Elsevier and Springer XML parsing
* Figure and caption extraction
* Scientific figure panel detection
* Google Gemini and Azure OpenAI caption generation
* Dataset construction for multimodal machine learning

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   getting_started
   examples
   api
