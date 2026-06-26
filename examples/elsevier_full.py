from matmmextract.preprocess.pipeline import load_csvs
from matmmextract.inference.detector import detect
from matmmextract.inference.cropper import crop
from matmmextract.inference.crop_csv_builder import build_crop_csv
from matmmextract.inference.captioner_azure import captioner as azure_caption
from matmmextract.inference.dataset_builder import build
from matmmextract.openalex.fetcher import fetch_elsevier
from matmmextract.elsevier.extractor import extract_all as elsevier_extract
from matmmextract.elsevier.downloader import download_all as elsevier_download
from matmmextract.elsevier.fetcher import fetch_all as elsevier_fetch


elsevier_result = fetch_elsevier(
    license_="cc-by",
    keywords=["titanium"],
    from_year=2020, to_year=2024,
    max_results=1,
    output_csv="output/elsevier_papers.csv",
    api_key="",
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
    checkpoint="https://drive.google.com/file/d/10garsNWEdgzMGX9nyDE8dMABkU_3BYp9/view?usp=sharing",
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
    azure_endpoint="",
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

