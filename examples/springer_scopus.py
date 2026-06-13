from multimat.preprocess import (
    load_csvs,
    drop_duplicate_dois,
    save_csv,
)

from multimat.springer import (
    fetch_all as springer_fetch,
    extract_all as springer_extract,
    download_all as springer_download,
)


# ── Step 1: Load CSVs ────────────────────────────────────────────────────────
springer_df = load_csvs(["scopus.csv"])
springer_df = drop_duplicate_dois(springer_df)

springer_df = filter_by_publisher(df, pattern=r"\bSpringer\b")

save_csv(springer_df, "output/springer_papers.csv")


# ── Step 2: Fetch Springer XMLs ──────────────────────────────────────────────
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