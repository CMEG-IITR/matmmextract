from paper_pipeline.preprocess.pipeline import load_csvs
from paper_pipeline.inference.detector import detect
from paper_pipeline.inference.cropper import crop
from paper_pipeline.inference.crop_csv_builder import build_crop_csv
from paper_pipeline.inference.captioner_azure import captioner as azure_caption
from paper_pipeline.inference.dataset_builder import build
from paper_pipeline.openalex.fetcher import fetch_elsevier
from paper_pipeline.elsevier.extractor import extract_all as elsevier_extract
from paper_pipeline.elsevier.downloader import download_all as elsevier_download
from paper_pipeline.elsevier.fetcher import fetch_all as elsevier_fetch


elsevier_result = fetch_elsevier(
    license_="cc-by",
    keywords=["titanium alloy", "microstructure"],
    from_year=2020, to_year=2024,
    max_results=4,
    output_csv="output/elsevier_papers.csv",
    email="mohammad_i@mfs.iitr.ac.in",   # optional, gets faster rate limit
)

elsevier_df = load_csvs(["output/elsevier_papers.csv"])

elsevier_fetch(
    df=elsevier_df,
    api_key="",
    inst_token="",
    output_dir="_elsevier",
)

elsevier_figs, _ = elsevier_extract(
    "_elsevier",
    output_csv="output/elsevier_figures.csv",
)

elsevier_download(
    csv_path="output/elsevier_figures.csv",
    output_dir="images/elsevier",
    api_key="",
    inst_token="",
)


# Step 8: detect
detect(
    image_dir="images/elsevier",
    output_dir="inference_results",
    checkpoint="models/best.pt",
    conf=0.6, iou=0.4, imgsz=1024,
)

# Step 9: crop
crop(
    image_dir="images/elsevier",
    json_dir="inference_results",
    output_dir="crops",
)

# Step 9.5: build captioning CSV
build_crop_csv(
    crops_dir="crops",
    figures_csv="output/elsevier_figures.csv",
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
    output_csv="elsevier_linked_dataset.csv",
)
