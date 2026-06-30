<p align="center">
  <img src="https://raw.githubusercontent.com/CMEG-IITR/matmmextract/release/logo.svg" alt="MatMMExtract Logo" width="220">
</p>

# MatmmExtract

**MatmmExtract** is an end-to-end pipeline for building multimodal materials-science datasets from scientific literature.

Starting from OpenAlex or Scopus metadata, MatmmExtract automatically retrieves papers, extracts figures and captions, downloads images, detects sub-panels, generates fine-grained captions using modern LLMs, and links everything into a machine-learning-ready dataset.

---

## Features

### Literature Acquisition

- OpenAlex search integration

- Elsevier full-text retrieval

- Springer full-text retrieval

- Scopus CSV workflow support

- Open-access filtering

- DOI deduplication

### Figure Extraction

- Parse Elsevier XML articles

- Parse Springer XML articles

- Extract figure metadata

- Extract captions

- Extract figure reference sentences

- Preserve paper-level metadata

### License Processing

- Detect Creative Commons licenses

- Filter CC-BY content

- Generate license audit reports

- Support large-scale corpus filtering

### Image Downloading

- Download publisher-hosted figures

- Retry and resume support

- Download logging

- Filename normalization

### Vision Pipeline

- Scientific figure panel detection

- Automatic crop generation

- Crop-to-figure linking

- Dataset preparation utilities

### Caption Generation

- Gemini support

- Azure OpenAI support

- Batch captioning workflows

- Rate-limited API execution

- Structured JSON outputs

### Dataset Construction

- Link crops, captions, figures, and metadata

- Generate training-ready CSV datasets

- Create multimodal instruction-tuning corpora

---

## Installation

```bash
pip install matmmextract
```

Or install from source:

```bash
git clone https://github.com/<your-org>/matmmextract.git
cd matmmextract

pip install -e .
```

---

## Quick Start

### Search OpenAlex

```python
from matmmextract.openalex import fetch_elsevier

fetch_elsevier(
    keywords=["titanium alloy", "microstructure"],
    license_="cc-by",
    from_year=2020,
    to_year=2024,
    max_results=100,
    output_csv="papers.csv",
)
```

---

### Fetch Elsevier XMLs

```python
from matmmextract.elsevier import fetch_all

fetch_all(
    df=papers_df,
    api_key="YOUR_API_KEY",
    inst_token="YOUR_INST_TOKEN",
    output_dir="xmls",
)
```

---

### Detect Panels

```python
from matmmextract.inference import detect

detect(
    image_dir="images",
    output_dir="detections",
    checkpoint="best.pt",
)
```

---

### Generate Captions

```python
from matmmextract.inference import gemini_captioner

gemini_captioner(
    csv_path="crops.csv",
    output_dir="subcaptions",
    api_key="YOUR_API_KEY",
)
```

---

### Build Final Dataset

```python
from matmmextract.inference import build

build(
    images_dir="crops",
    json_dir="subcaptions",
    output_csv="linked_dataset.csv",
)
```

---

## Example Pipelines

MatmmExtract ships with complete examples:

```text
examples/
├── elsevier_full.py
├── elsevier_scopus.py
├── springer_full.py
└── springer_scopus.py
```

### OpenAlex → Elsevier → Azure → Dataset

```bash
python examples/elsevier_full.py
```

### Scopus → Elsevier → Azure → Dataset

```bash
python examples/elsevier_scopus.py
```

### OpenAlex → Springer → Gemini → Dataset

```bash
python examples/springer_full.py
```

### Scopus → Springer → Azure → Dataset

```bash
python examples/springer_scopus.py
```

---

## Package Structure

```text
matmmextract
├── openalex
├── elsevier
├── springer
├── preprocess
├── inference
└── shared
```

### OpenAlex

Paper discovery and metadata retrieval.

### Elsevier

Full-text retrieval, figure extraction, and image downloading.

### Springer

Full-text retrieval, figure extraction, and image downloading.

### Preprocess

Dataset filtering, DOI processing, publisher filtering, and license analysis.

### Inference

Detection, cropping, caption generation, and dataset construction.

### Shared

Common utilities used throughout the pipeline.

---

## Documentation

Build locally:

```bash
sphinx-build -b html docs docs/_build
```

Generated documentation:

```text
docs/_build/index.html
```

---

## Citation

If you use MatmmExtract in academic work, please cite:

```text
MatmmExtract: A Pipeline for Constructing Multimodal Materials-Science Datasets
from Scientific Literature.
```

---

## License

GNU General Public License v3.0 (GPL-3.0).

See the LICENSE file for details.

---

## Authors

- Subham Ghosh

- Abhishek Tewari

- Mohammad Ibrahim
