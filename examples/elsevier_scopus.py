from multimat.preprocess import (
    load_csvs,
    drop_duplicate_dois,
    filter_open_access,
    save_csv,
)

from multimat.preprocess import (
    scan_directory,
    filter_figures_cc_by,
)

from multimat.elsevier import (
    fetch_all as elsevier_fetch,
    extract_all as elsevier_extract,
    download_all as elsevier_download,
)


# ── Step 1: Load CSVs ────────────────────────────────────────────────────────
elsevier_df = load_csvs(["scopus.csv"])
elsevier_df = drop_duplicate_dois(elsevier_df)

elsevier_oa = filter_open_access(elsevier_df)

save_csv(elsevier_df, "output/elsevier_papers.csv")


# ── Step 2: Fetch Elsevier XMLs ──────────────────────────────────────────────
elsevier_fetch(
    df=elsevier_df,
    api_key="",
    inst_token="",
    output_dir="_elsevier",
)


# ── Step 3: Extract figures ──────────────────────────────────────────────────
elsevier_figs, _ = elsevier_extract(
    "_elsevier",
    output_csv="output/elsevier_figures.csv",
)


# ── Step 4: CC-BY filtering ──────────────────────────────────────────────────
cc_df, _ = scan_directory(
    "_elsevier",
    output_csv="output/elsevier_cc.csv",
)

elsevier_figs_ccby = filter_figures_cc_by(
    elsevier_figs,
    cc_df,
    output_csv="output/elsevier_figures_ccby.csv",
)


# ── Step 5: Download images ──────────────────────────────────────────────────
elsevier_download(
    csv_path="output/elsevier_figures_ccby.csv",
    output_dir="images/elsevier",
    api_key="",
    inst_token="",
)