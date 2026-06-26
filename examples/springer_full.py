from matmmextract.openalex import fetch_springer
from matmmextract.preprocess import load_csvs

from matmmextract.springer import (
    fetch_all as springer_fetch,
    extract_all as springer_extract,
    download_all as springer_download,
)

from matmmextract.inference import (
    detect,
    crop,
    build_crop_csv,
    gemini_captioner as gemini_caption,
    build,
)

springer_result = fetch_springer(
    license_=["cc-by", "cc-by-nc"],
    keywords=["alloy"],
    max_results=1,
    output_csv="output/springer_papers.csv",
    api_key="",
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
    checkpoint="https://drive.google.com/file/d/10garsNWEdgzMGX9nyDE8dMABkU_3BYp9/view?usp=sharing",
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

# Step 10: generate sub-captions via gemini
gemini_caption(
    csv_path="output/crops_for_captioning.csv",
    output_dir="subcaptions",
    api_key="",
)

# Step 11: link crops + captions → final dataset
build(
    images_dir="crops",
    json_dir="subcaptions",
    output_csv="springer_linked_dataset.csv",
)