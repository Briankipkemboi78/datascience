import pandas as pd
import numpy as np
from fact_block_etl import FactBlockETL
from utils import log_event, BLANK_SET, is_selected, clean_str


def transform_gap_pruning(df, column_map):
    df = df.rename(columns=column_map)

    # Updated score map
    score_map = {"good": 1, "medium": 2, "bad": 3}

    condition_cols = [
        "trunk_crown_1", "trunk_crown_2", "trunk_crown_3",
        "chupons_1", "chupons_2", "chupons_3",
        "dead_branch_1", "dead_branch_2", "dead_branch_3",
        "secondary_branch_1", "secondary_branch_2", "secondary_branch_3"
    ]

    for c in condition_cols:
        if c in df:
            s = clean_str(df[c]).str.lower()
            df[c] = s.map(score_map).fillna(df[c])

    # Totals + Nestlé scores
    df["total_trunk_crown"] = df[["trunk_crown_1", "trunk_crown_2", "trunk_crown_3"]].sum(axis=1, min_count=1)
    df["trunk_crown_nestle"] = (df["total_trunk_crown"] / 3).round()

    df["total_chupons"] = df[["chupons_1", "chupons_2", "chupons_3"]].sum(axis=1, min_count=1)
    df["chupon_nestle"] = (df["total_chupons"] / 3).round()

    df["total_dead_branches"] = df[["dead_branch_1", "dead_branch_2", "dead_branch_3"]].sum(axis=1, min_count=1)
    df["dead_nestle"] = (df["total_dead_branches"] / 3).round()

    df["total_secondary_branches"] = df[["secondary_branch_1", "secondary_branch_2", "secondary_branch_3"]].sum(axis=1, min_count=1)
    df["secondary_nestle"] = (df["total_secondary_branches"] / 3).round()

    # --- BON flags ---
    df["trunccrown_bon"] = ((df["trunk_crown_nestle"] > 0.99) & (df["trunk_crown_nestle"] < 1.49)).astype(int)
    df["chupon_bon"] = ((df["chupon_nestle"] > 0.99) & (df["chupon_nestle"] < 1.49)).astype(int)
    df["dead_bon"] = ((df["dead_nestle"] > 0.99) & (df["dead_nestle"] < 1.49)).astype(int)
    df["secondary_bon"] = ((df["secondary_nestle"] > 0.99) & (df["secondary_nestle"] < 1.49)).astype(int)

    df["total_bon"] = df[["trunccrown_bon", "chupon_bon", "dead_bon", "secondary_bon"]].sum(axis=1)

    # --- MOYEN flags ---
    df["trunccrown_moyen"] = ((df["trunk_crown_nestle"] > 1.50) & (df["trunk_crown_nestle"] < 2.49)).astype(int)
    df["chupon_moyen"] = ((df["chupon_nestle"] > 1.50) & (df["chupon_nestle"] < 2.49)).astype(int)
    df["dead_moyen"] = ((df["dead_nestle"] > 1.50) & (df["dead_nestle"] < 2.49)).astype(int)
    df["secondary_moyen"] = ((df["secondary_nestle"] > 1.50) & (df["secondary_nestle"] < 2.49)).astype(int)

    df["total_moyen"] = df[["trunccrown_moyen", "chupon_moyen", "dead_moyen", "secondary_moyen"]].sum(axis=1)

    # --- Nestlé scoring (IFS logic) ---
    conditions = [
        (df["total_bon"] == 4),
        (df["total_bon"] == 2) & (df["total_moyen"] == 2),
        (df["total_bon"] == 3) & (df["total_moyen"] == 1),
        (df["total_bon"] >= 1) & (df["total_moyen"] >= 1),
    ]
    choices = [1, 2, 2, 3]
    default = 4
    df["nestle_scoring"] = np.select(conditions, choices, default=default)

    # --- Nestlé pass/fail flag ---
    df["nestle_pass_fail"] = np.where(df["nestle_scoring"].isin([1, 2]), 1, 0)


    keep = [
        "result_id",
        "trunk_crown_1", "trunk_crown_2", "trunk_crown_3", "total_trunk_crown", "trunk_crown_nestle",
        "chupons_1", "chupons_2", "chupons_3", "total_chupons", "chupon_nestle",
        "dead_branch_1", "dead_branch_2", "dead_branch_3", "total_dead_branches", "dead_nestle",
        "secondary_branch_1", "secondary_branch_2", "secondary_branch_3", "total_secondary_branches", "secondary_nestle",
        "trunccrown_bon", "chupon_bon", "dead_bon", "secondary_bon", "total_bon",
        "trunccrown_moyen", "chupon_moyen", "dead_moyen", "secondary_moyen", "total_moyen",
        "nestle_scoring", "nestle_pass_fail"
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep]


def fact_gap_pruning():
    column_map = {
        "result_id": "result_id",

        # Trunk & Crown
        "Trunk & Crown": "trunk_crown_1",
        "Trunk & Crown.1": "trunk_crown_2",
        "Trunk & Crown.2": "trunk_crown_3",

        # Chupons
        "Chupons": "chupons_1",
        "Chupons.1": "chupons_2",
        "Chupons.2": "chupons_3",

        # Dead Branches
        "Dead Branches": "dead_branch_1",
        "Dead Branches.1": "dead_branch_2",
        "Dead Branches.2": "dead_branch_3",

        # Secondary Branches
        "Secondary Branches": "secondary_branch_1",
        "Secondary Branches.1": "secondary_branch_2",
        "Secondary Branches.2": "secondary_branch_3",
    }

    etl = FactBlockETL(block_name="fact_gap_pruning", column_map=column_map)
    etl.transform = lambda df: transform_gap_pruning(df, column_map)
    etl.run()


if __name__ == "__main__":
    fact_gap_pruning()
