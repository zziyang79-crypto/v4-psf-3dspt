"""Consistent CSV loading for the production pipeline.

The pipeline works on whole DataFrames after loading.  Letting pandas infer
each file in chunks therefore adds a second dtype reconciliation pass and can
emit DtypeWarning when sparse diagnostic text columns start with empty cells.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


# These columns are categorical/audit text by contract.  Their empty values
# must not cause a later non-empty chunk to be reinterpreted as a new dtype.
TEXT_COLUMNS = frozenset(
    {
        "Angle_Intensity_Message",
        "Angle_Intensity_Source",
        "Angle_Length_QC_Reason",
        "Assigned_Source_ID",
        "Association_Status",
        "Center_Method",
        "Cluster_Type",
        "Cross_Parent_Overlap_Repair_Source",
        "Detection_Selection_Method",
        "Geometry_Repair_Source",
        "Geometry_Subspot_Repair_Source",
        "Localization_Message",
        "Localization_Method",
        "Member_Track_IDs",
        "Source_Cluster_IDs",
        "Source_Clusters",
        "Subspot_Completion_Note",
        "Subspot_Position_Source",
        "Theta_Tracked_Simple_Average_Source",
        "Trajectory_Method",
        "XY_Localization_Message",
        "XY_Localization_Method",
        "Z_Angle_Out_Of_Domain_Level",
        "Z_Final_Source",
        "Z_Localization_Message",
        "Z_Localization_Method",
        "Z_QC_Group_Basis",
        "Z_QC_Group_ID",
        "Z_Confidence",
        "lap_link_status",
        "possible_particle_ids",
        "possible_source_ids",
        "source_id",
    }
)


def read_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Load one complete pipeline CSV with stable audit-column dtypes.

    All production callers consume a full DataFrame, so ``low_memory=False``
    avoids chunk-wise dtype inference and its reconciliation overhead.  The
    small explicit string map additionally protects sparse diagnostic columns
    whose first non-empty values may appear late in a large file.
    """

    supplied_dtype = kwargs.get("dtype")
    if supplied_dtype is None:
        dtype: dict[str, Any] = {}
    elif isinstance(supplied_dtype, dict):
        dtype = dict(supplied_dtype)
    else:
        # A caller that deliberately requests one dtype for every column owns
        # that decision; do not replace it with the audit-column map.
        dtype = {}

    if isinstance(supplied_dtype, dict) or supplied_dtype is None:
        for column in TEXT_COLUMNS:
            # ``object`` keeps the C parser's fast string path.  Pandas'
            # nullable ``string`` extension array is semantically fine here,
            # but adds an avoidable conversion for large audit tables.
            dtype.setdefault(column, "object")
        kwargs["dtype"] = dtype

    # The downstream algorithms need all rows at once, so chunked inference
    # cannot reduce their peak memory.  A single inference pass is faster and
    # prevents DtypeWarning for late sparse diagnostic text.
    kwargs["low_memory"] = False
    return pd.read_csv(path, **kwargs)


def find_header_row(
    path: str | Path,
    required_columns: Iterable[str],
    *,
    max_rows: int = 6,
    encoding: str = "utf-8-sig",
) -> int:
    """Find a likely header line without repeatedly parsing the whole CSV."""

    targets = {str(value).strip().lower() for value in required_columns}
    try:
        with Path(path).open("r", encoding=encoding, errors="replace", newline="") as handle:
            for row_index, row in enumerate(csv.reader(handle)):
                if row_index >= max_rows:
                    break
                names = {str(value).strip().lower() for value in row}
                if targets & names:
                    return row_index
    except OSError:
        pass
    return 0
