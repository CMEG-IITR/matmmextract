from paper_pipeline.openalex.fetcher import fetch_springer
from paper_pipeline.preprocess.pipeline import load_csvs

from paper_pipeline.springer.fetcher import fetch_all as springer_fetch
from paper_pipeline.springer.extractor import extract_all as springer_extract
from paper_pipeline.springer.downloader import download_all as springer_download

from paper_pipeline.inference.detector import detect
from paper_pipeline.inference.cropper import crop
from paper_pipeline.inference.crop_csv_builder import build_crop_csv
from paper_pipeline.inference.captioner_azure import captioner as azure_caption
from paper_pipeline.inference.dataset_builder import build

springer_result = fetch_springer(
    license_="cc-by",
    keywords=["alloy"],
    max_results=2,
    output_csv="output/springer_papers.csv",
)

springer_df = load_csvs(["output/springer_papers.csv"])

springer_fetch(
    df=springer_df,
    api_key="",
    output_dir="_springer",
    use_open_access=True,
)


# ── Step 3: Extract figures from XMLs ────────────────────────────────────────
springer_figs, _ = springer_extract(
    "_springer",
    output_csv="output/springer_figures.csv",
)


# ── Step 4: Download figure images ───────────────────────────────────────────
springer_download(
    csv_path="output/springer_figures.csv",
    output_dir="images/springer",
)

detect(
    image_dir="images/springer",
    output_dir="inference_results",
    checkpoint="models/best.pt",
    conf=0.6, iou=0.4, imgsz=1024,
)

# Step 9: crop
crop(
    image_dir="images/springer",
    json_dir="inference_results",
    output_dir="crops",
)

# Step 9.5: build captioning CSV
build_crop_csv(
    crops_dir="crops",
    figures_csv="output/springer_figures.csv",
    output_csv="output/crops_for_captioning.csv",
)

# Step 10: generate sub-captions via Azure
azure_caption(
    csv_path="output/crops_for_captioning.csv",
    output_dir="subcaptions",
    api_key="",
    model_name="Mistral-Large-3",
    image_name_col="downloaded_image_name",   # matches crops_for_captioning.csv
    caption_col="caption",
    reference_col="reference_sentences",
)

# Step 11: link crops + captions → final dataset
build(
    images_dir="crops",
    json_dir="subcaptions",
    output_csv="springer_linked_dataset.csv",
)