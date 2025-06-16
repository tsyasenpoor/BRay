"""Utilities for analyzing pathway based experiments.

This module contains a collection of helper functions that were
originally scattered through the ``path_analysis.ipynb`` notebook.
The goal of this file is to provide a cleaner interface for loading
experiment results and performing common analyses.
"""

from __future__ import annotations

import gzip
import json
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, roc_curve


# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------

def load_experiment_data(instance_path: str) -> Tuple[pd.DataFrame | None, pd.DataFrame | None,
                                                    pd.DataFrame | None, pd.DataFrame | None,
                                                    pd.DataFrame | None]:
    """Load a single experiment directory.

    Parameters
    ----------
    instance_path:
        Path to a directory containing ``*_analysis.json.gz`` and
        ``*_results.json.gz`` files as well as ``*_train.csv.gz`` etc.
    Returns
    -------
    Tuple of (pathways, statistics, train_stats, test_stats, val_stats).
    Each entry is a :class:`pandas.DataFrame` or ``None`` if not found.
    """
    dfs = {
        "pathways": None,
        "statistics": None,
        "train_stats": None,
        "test_stats": None,
        "val_stats": None,
    }

    if not os.path.isdir(instance_path):
        print(f"Directory not found: {instance_path}")
        return tuple(dfs.values())

    for filename in os.listdir(instance_path):
        fp = os.path.join(instance_path, filename)
        if not os.path.isfile(fp):
            continue
        try:
            if "analysis" in filename and filename.endswith(".json.gz"):
                with gzip.open(fp, "rt", encoding="utf-8") as f:
                    dfs["pathways"] = pd.DataFrame(json.load(f))
            elif "results" in filename and filename.endswith(".json.gz"):
                with gzip.open(fp, "rt", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    dfs["statistics"] = pd.DataFrame([data])
                else:
                    dfs["statistics"] = pd.DataFrame(data)
            elif filename.endswith("train.csv.gz"):
                dfs["train_stats"] = pd.read_csv(fp, compression="gzip")
            elif filename.endswith("test.csv.gz"):
                dfs["test_stats"] = pd.read_csv(fp, compression="gzip")
            elif filename.endswith("val.csv.gz"):
                dfs["val_stats"] = pd.read_csv(fp, compression="gzip")
        except Exception as exc:  # noqa: BLE001
            print(f"Error processing {filename}: {exc}")

    return (dfs["pathways"], dfs["statistics"], dfs["train_stats"],
            dfs["test_stats"], dfs["val_stats"])


def analyze_statistics(statistics_df: pd.DataFrame) -> Tuple[List[Dict[str, float]], pd.DataFrame]:
    """Return metric summaries from a statistics DataFrame."""
    row = statistics_df.iloc[0]
    train = row["train_metrics"]
    test = row["test_metrics"]
    val = row["val_metrics"]

    metrics_list = [
        {"set": "train", "metrics": train},
        {"set": "validation", "metrics": val},
        {"set": "test", "metrics": test},
    ]
    metrics_df = pd.DataFrame([
        {"Set": "Train", **train},
        {"Set": "Validation", **val},
        {"Set": "Test", **test},
    ])
    return metrics_list, metrics_df


# ---------------------------------------------------------------------------
# Probability group helpers
# ---------------------------------------------------------------------------

def split_probability_groups(df: pd.DataFrame, *, low: float = 0.2, high: float = 0.8) -> pd.DataFrame:
    """Return a DataFrame with samples labelled as ``low`` or ``high`` probability."""
    low_df = df[df["probability"] < low].copy()
    low_df["group"] = "low"
    high_df = df[df["probability"] >= high].copy()
    high_df["group"] = "high"
    return pd.concat([low_df, high_df])


# ---------------------------------------------------------------------------
# Gene program utilities
# ---------------------------------------------------------------------------

def build_gene_program_matrix(pathways: pd.DataFrame) -> pd.DataFrame:
    """Create a gene program weight matrix from pathway analysis output.

    The input ``pathways`` DataFrame must contain a ``gene_programs`` column
    with dictionaries that include ``genes`` and ``upsilon``.
    """
    records: List[Dict[str, float]] = []
    for program_name, row in pathways.iterrows():
        gp = row.get("gene_programs")
        if not isinstance(gp, dict) or "genes" not in gp:
            continue
        entry = {"program_name": program_name, "upsilon": gp.get("upsilon", np.nan)}
        for gene in gp["genes"]:
            if "gene" in gene and "weight" in gene:
                entry[gene["gene"]] = gene["weight"]
        records.append(entry)

    df = pd.DataFrame(records)
    if "program_name" in df.columns:
        df = df.set_index("program_name")
    df = df.fillna(0)
    return df


def calculate_weighted_pathway_similarity(
    gene_program_df: pd.DataFrame,
    pathway_data: Dict[str, Iterable[str]],
    gene_columns: Iterable[str],
    top_n: int = 50,
) -> pd.DataFrame:
    """Compute simple similarity metrics between programs and pathways."""
    results = []
    pathway_keys = list(pathway_data.keys())
    for i, program in enumerate(gene_program_df.index):
        if i >= len(pathway_keys):
            break
        pathway_key = pathway_keys[i]
        pathway_genes = set(pathway_data[pathway_key])
        weights = gene_program_df.loc[program, list(gene_columns)]

        total_weight = np.sum(np.abs(weights))
        pathway_weight = np.sum(np.abs(weights[weights.index.isin(pathway_genes)]))
        weight_concentration = (pathway_weight / total_weight) * 100 if total_weight > 0 else 0

        top_genes = weights.abs().sort_values(ascending=False).head(top_n).index
        top_in_pathway = sum(g in pathway_genes for g in top_genes) / len(top_genes) * 100

        results.append(
            {
                "Program": program,
                "Original_Pathway": pathway_key,
                "Weight_Concentration": weight_concentration,
                "Top{}_In_Pathway".format(top_n): top_in_pathway,
                "Upsilon": gene_program_df.loc[program, "upsilon"] if "upsilon" in gene_program_df.columns else np.nan,
            }
        )
    return pd.DataFrame(results)


def top_genes_per_program(
    gene_program_df: pd.DataFrame,
    *,
    top_n: int = 100,
    gene_columns: Iterable[str] | None = None,
) -> Dict[str, List[str]]:
    """Return the most informative genes for each gene program.

    Parameters
    ----------
    gene_program_df:
        DataFrame produced by :func:`build_gene_program_matrix` with gene weights.
    top_n:
        Number of genes to retain per program based on absolute weight.
    gene_columns:
        Optional list of columns representing gene weights. If ``None``, all
        columns except ``upsilon`` are used.

    Returns
    -------
    Dict[str, List[str]]
        Mapping of program name to list of selected genes.
    """
    if gene_columns is None:
        gene_columns = [c for c in gene_program_df.columns if c != "upsilon"]

    top_genes: Dict[str, List[str]] = {}
    for program in gene_program_df.index:
        weights = gene_program_df.loc[program, gene_columns].abs()
        genes = list(weights.sort_values(ascending=False).head(top_n).index)
        top_genes[program] = genes

    return top_genes


def hypergeometric_enrichment(
    group_genes: Iterable[str],
    program_genes: Iterable[str],
    background_genes: Iterable[str],
) -> float:
    """Return hypergeometric p-value of overlap between two gene sets."""
    background_genes = set(background_genes)
    group_genes = set(group_genes) & background_genes
    program_genes = set(program_genes) & background_genes

    overlap = len(group_genes & program_genes)
    M = len(background_genes)
    n = len(program_genes)
    N = len(group_genes)
    from scipy.stats import hypergeom

    return float(hypergeom.sf(overlap - 1, M, n, N))


def gene_program_enrichment(
    groups: Dict[str, Iterable[str]],
    program_gene_sets: Dict[str, Iterable[str]],
    background_genes: Iterable[str],
) -> pd.DataFrame:
    """Compute enrichment of each program within provided gene groups."""
    records = []
    bg = set(background_genes)
    for group_name, genes in groups.items():
        for program_name, program_genes in program_gene_sets.items():
            pval = hypergeometric_enrichment(genes, program_genes, bg)
            overlap = len(set(genes) & set(program_genes))
            records.append(
                {
                    "Group": group_name,
                    "Program": program_name,
                    "Overlap": overlap,
                    "PValue": pval,
                }
            )

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Benjamini-Hochberg FDR
    df = df.sort_values("PValue").reset_index(drop=True)
    n = len(df)
    bh_values = []
    prev_q = 1.0
    for i, p in enumerate(df["PValue"], 1):
        q = min(prev_q, p * n / i)
        bh_values.append(q)
        prev_q = q
    df["FDR"] = bh_values
    return df

