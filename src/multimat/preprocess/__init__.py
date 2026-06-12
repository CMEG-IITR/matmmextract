from .pipeline import (
    load_csvs,
    publisher_counts,
    find_duplicate_dois,
    drop_duplicate_dois,
    filter_by_publisher,
    filter_open_access,
    save_csv,
    move_xmls_by_doi,
    copy_xmls_by_filename,
    flatten_images,
    prune_images_without_captions,
    run,
)

from .cc_license import (
    analyse_file,
    scan_directory,
    filter_figures_cc_by,
)

__all__ = [
    "load_csvs",
    "publisher_counts",
    "find_duplicate_dois",
    "drop_duplicate_dois",
    "filter_by_publisher",
    "filter_open_access",
    "save_csv",
    "move_xmls_by_doi",
    "copy_xmls_by_filename",
    "flatten_images",
    "prune_images_without_captions",
    "run",
    "analyse_file",
    "scan_directory",
    "filter_figures_cc_by",
]
