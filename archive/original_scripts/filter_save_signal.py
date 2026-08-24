# -*- coding: utf-8 -*-
"""
Combined DNN+GNN model for high-energy physics classification.

This script loads pre-processed event data (leptons, jets, fat jets, MET)
and fat jet constituent data.

It uses a GNN (Interaction Network) to process the constituents of the
leading fat jet, creating a learned representation. This representation
is then combined with high-level event features (lepton kinematics,
other jet kinematics, MET) and fed into a Deep Neural Network (DNN)
for binary classification (signal vs. background).
"""

"""
## Setup
"""

import pandas as pd
import matplotlib.pyplot as plt

# --- Plot Helpers (from original script) ---

def plot_histogram(df, column, bins=50, title=None, xlabel=None, ylabel="Frequency", yscale ='linear'):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    plt.figure()
    plt.hist(df[column], bins=bins, edgecolor='black')
    plt.title(title or f"Histogram of {column}")
    plt.xlabel(xlabel or column)
    plt.ylabel(ylabel)
    plt.yscale(yscale)
    plt.grid(True)
    plt.show()

def plot_histogram_by_type(df, value_column, type_column="type", bins=50, alpha=0.7, filename = 'default_file', title = None):
    if value_column not in df.columns or type_column not in df.columns:
        raise ValueError(f"Columns '{value_column}' and/or '{type_column}' not found in DataFrame.")

    # Compute shared bin edges from entire column
    data = df[value_column].dropna()
    bin_edges = np.linspace(data.min(), data.max(), bins + 1)

    plt.figure(figsize=(8, 6))

    for t in df[type_column].unique():
        subset = df[df[type_column] == t][value_column].dropna()
        plt.hist(subset, bins=bin_edges, alpha=alpha, label=t, edgecolor='black', histtype='stepfilled')

    plt.xlabel(value_column)
    plt.ylabel("Frequency")
    plt.yscale('log')
    title = title or f"Histogram of {value_column} by {type_column} from {filename}"
    plt.title(title)
    plt.legend(title=type_column)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Physics Helper Functions (from original script) ---

def delta_phi(phi1, phi2):
    """Computes the difference in phi, handling the wrap-around at pi."""
    # Ensure inputs are numpy arrays for broadcasting
    dphi = np.asarray(phi1) - np.asarray(phi2)
    dphi = np.where(dphi > np.pi, dphi - 2 * np.pi, dphi)
    dphi = np.where(dphi < -np.pi, dphi + 2 * np.pi, dphi)
    return dphi

def delta_r(eta1, eta2, phi1, phi2):
    """Computes the delta R distance between two objects."""
    deta = np.asarray(eta1) - np.asarray(eta2)
    dphi = delta_phi(phi1, phi2)
    return np.sqrt(deta**2 + dphi**2)

path_name = "data/"

# Signal
signal_df_lep = pd.read_csv(path_name + "signal_df_lep.csv")
signal_df_g = pd.read_csv(path_name + "signal_df_g.csv")
signal_df_j = pd.read_csv(path_name + "signal_df_j_cleaned.csv")
signal_df_fj = pd.read_csv(path_name + "signal_df_fj_cleaned.csv")
signal_df_met = pd.read_csv(path_name + "signal_df_met.csv")
signal_df_fjc = pd.read_csv(path_name + "signal_df_fjc.csv")

# Background 280
background_280_df_lep = pd.read_csv(path_name + "background_280_df_lep.csv")
background_280_df_j = pd.read_csv(path_name + "background_280_df_j_cleaned.csv")
background_280_df_fj = pd.read_csv(path_name + "background_280_df_fj_cleaned.csv")
background_280_df_met = pd.read_csv(path_name + "background_280_df_met.csv")
background_280_df_fjc = pd.read_csv(path_name + "background_280_df_fjc.csv")

# Background 500
background_500_df_lep = pd.read_csv(path_name + "background_500_df_lep.csv")
background_500_df_j = pd.read_csv(path_name + "background_500_df_j_cleaned.csv")
background_500_df_fj = pd.read_csv(path_name + "background_500_df_fj_cleaned.csv")
background_500_df_met = pd.read_csv(path_name + "background_500_df_met.csv")
background_500_df_fjc = pd.read_csv(path_name + "background_500_df_fjc.csv")

# Background 1000
background_1000_df_lep = pd.read_csv(path_name + "background_1000_df_lep.csv")
background_1000_df_j = pd.read_csv(path_name + "background_1000_df_j_cleaned.csv")
background_1000_df_fj = pd.read_csv(path_name + "background_1000_df_fj_cleaned.csv")
background_1000_df_met = pd.read_csv(path_name + "background_1000_df_met.csv")
background_1000_df_fjc = pd.read_csv(path_name + "background_1000_df_fjc.csv")

# -*- coding: utf-8 -*-
"""
Step 1: Physics Selection & Filtering (Parquet)

Logic:
1. Identify valid events (Must have lepton + fat jet).
2. Select exactly ONE fat jet per event (closest to lepton).
3. Filter all other dataframes to these events.
4. Save as Parquet in original 'long' format (no pivoting yet).
"""

import os
import numpy as np
import pandas as pd

# --- Setup Output Directory ---
OUTPUT_DIR = 'data_filtered'
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Saving processed files to: {OUTPUT_DIR}/")


# --- Helper Functions ---

def delta_phi(phi1, phi2):
    dphi = np.asarray(phi1) - np.asarray(phi2)
    dphi = np.where(dphi > np.pi, dphi - 2 * np.pi, dphi)
    dphi = np.where(dphi < -np.pi, dphi + 2 * np.pi, dphi)
    return dphi


def delta_r(eta1, eta2, phi1, phi2):
    deta = np.asarray(eta1) - np.asarray(eta2)
    dphi = delta_phi(phi1, phi2)
    return np.sqrt(deta ** 2 + dphi ** 2)


def filter_and_save(name, df_lep, df_j, df_fj, df_met, df_fjc, df_gen=None):
    print(f"\nProcessing {name}...")

    # 1. Identify Common Valid Events
    # Must have at least 1 lepton and 1 fat jet
    valid_lep_idx = df_lep['event_index'].unique()
    valid_fj_idx = df_fj['event_index'].unique()

    # Intersection
    valid_events = np.intersect1d(valid_lep_idx, valid_fj_idx)
    print(
        f"  - Original events: {len(valid_lep_idx)} (lep) / {len(valid_fj_idx)} (fj)")
    print(f"  - Valid intersection: {len(valid_events)}")

    # 2. Filter Lepton (Keep all valid events, leading lepton only usually desired,
    # but user said "drop bad events", implying we keep the structure.
    # We'll filter to valid events. If there are multiple leptons, we keep them
    # (unless you strictly want 1). Assuming 1 per event based on previous code.)
    df_lep_filtered = df_lep[df_lep['event_index'].isin(valid_events)].copy()

    # To do the dR calculation, we need the leading lepton info temporarily
    # Sort by pT desc to get leading lepton
    df_lep_lead = df_lep_filtered.sort_values(['event_index', 'pt'],
                                              ascending=[True, False])
    df_lep_lead = \
    df_lep_lead.groupby('event_index').head(1).set_index('event_index')[
        ['eta', 'phi']]
    df_lep_lead = df_lep_lead.rename(columns={'eta': 'l_eta', 'phi': 'l_phi'})

    # 3. Select ONE Fat Jet (Closest to Lepton)
    df_fj_filtered = df_fj[df_fj['event_index'].isin(valid_events)].copy()

    # Merge lepton info
    df_fj_calc = df_fj_filtered.merge(df_lep_lead, left_on='event_index',
                                      right_index=True)

    # Calculate dR
    df_fj_calc['dr_lep'] = delta_r(df_fj_calc['eta'], df_fj_calc['l_eta'],
                                   df_fj_calc['phi'], df_fj_calc['l_phi'])

    # Sort by dR ascending and take head(1)
    df_fj_selected = df_fj_calc.sort_values(['event_index', 'dr_lep']).groupby(
        'event_index').head(1)

    # Drop the temp calc columns to restore original format
    df_fj_final = df_fj_selected.drop(columns=['l_eta', 'l_phi', 'dr_lep'])

    # 4. Filter Constituents (_fjc)
    # We only want constituents that belong to the selected fat jet.
    # Inner join on event_index AND 'ind' (which links to df_fj 'index')

    # We assume df_fj has a column 'index' that maps to df_fjc['ind'].
    # If df_fj uses a different column name for its ID, update 'right_on'.
    if 'index' not in df_fj_final.columns:
        # Fallback if 'index' is missing: assume 'ind' in fjc refers to rank 0 if we selected rank 0?
        # But since we selected by dR, the rank varies.
        # We MUST have the linking ID. Assuming input has 'index'.
        raise ValueError(
            "df_fj missing 'index' column required to link constituents.")

    keys = df_fj_final[['event_index', 'index']]
    df_fjc_final = df_fjc.merge(keys, left_on=['event_index', 'ind'],
                                right_on=['event_index', 'index'], how='inner')

    # Clean up merge artifacts if any (pandas might duplicate keys if names match, but here they match)
    # Ensure we don't have duplicate columns
    df_fjc_final = df_fjc_final.loc[:, ~df_fjc_final.columns.duplicated()]

    # 5. Filter Jets (_j) and MET (_met)
    # Simply keep all rows corresponding to the valid events
    df_j_final = df_j[df_j['event_index'].isin(valid_events)].copy()
    df_met_final = df_met[df_met['event_index'].isin(valid_events)].copy()

    # 6. (Optional) Signal Specific: Higgs matching
    # If this is signal, we might want to verify the Higgs is inside the jet.
    # But the prompt says "drop bad events e.g. if a event does not have any lepton".
    # It didn't strictly say "drop events where Higgs isn't in jet" for this step.
    # However, to be consistent with previous logic, if you want that cut, apply it here.
    # I will skip the geometric Higgs cut for now to strictly follow "save as how they are read but filtered".
    # The ML model handles training cuts.

    # 7. Save to Parquet
    print(f"  - Saving {len(df_lep_filtered)} lepton rows")
    df_lep_filtered.to_parquet(f"{OUTPUT_DIR}/{name}_df_lep.parquet")

    print(f"  - Saving {len(df_j_final)} jet rows")
    df_j_final.to_parquet(f"{OUTPUT_DIR}/{name}_df_j.parquet")

    print(f"  - Saving {len(df_fj_final)} fat jet rows (1 per event)")
    df_fj_final.to_parquet(f"{OUTPUT_DIR}/{name}_df_fj.parquet")

    print(f"  - Saving {len(df_met_final)} MET rows")
    df_met_final.to_parquet(f"{OUTPUT_DIR}/{name}_df_met.parquet")

    print(f"  - Saving {len(df_fjc_final)} constituent rows")
    df_fjc_final.to_parquet(f"{OUTPUT_DIR}/{name}_df_fjc.parquet")


def filter_and_save_signal(df_lep, df_j, df_fj, df_met, df_fjc, df_g):
    print("\nProcessing Signal (with Higgs Truth Matching)...")

    # 1. Identify Common Valid Events (Lepton + Fat Jet)
    valid_lep_idx = df_lep['event_index'].unique()
    valid_fj_idx = df_fj['event_index'].unique()
    valid_events = np.intersect1d(valid_lep_idx, valid_fj_idx)
    print(f"  - Initial valid events (Lep+FJ): {len(valid_events)}")

    # 2. Select ONE Fat Jet (Closest to Lepton)
    # We first filter FJ and Lep to the valid list to speed up calc
    df_fj_filtered = df_fj[df_fj['event_index'].isin(valid_events)].copy()
    df_lep_filtered = df_lep[df_lep['event_index'].isin(valid_events)].copy()

    # Get leading lepton coordinates
    df_lep_lead = df_lep_filtered.sort_values(['event_index', 'pt'],
                                              ascending=[True, False])
    df_lep_lead = \
    df_lep_lead.groupby('event_index').head(1).set_index('event_index')[
        ['eta', 'phi']]
    df_lep_lead = df_lep_lead.rename(columns={'eta': 'l_eta', 'phi': 'l_phi'})

    # Merge lepton info onto fat jets
    df_fj_calc = df_fj_filtered.merge(df_lep_lead, left_on='event_index',
                                      right_index=True)

    # Calculate dR(lep, fj)
    df_fj_calc['dr_lep'] = delta_r(df_fj_calc['eta'], df_fj_calc['l_eta'],
                                   df_fj_calc['phi'], df_fj_calc['l_phi'])

    # Select closest fat jet
    df_fj_selected = df_fj_calc.sort_values(['event_index', 'dr_lep']).groupby(
        'event_index').head(1)

    # 3. [CRITICAL] Higgs Truth Matching
    # Get True Higgs coordinates from df_g (id == 25)
    # We assume there is one Higgs per event in the signal file
    df_higgs = df_g[df_g['id'] == 25][['event_index', 'eta', 'phi']].set_index(
        'event_index')
    df_higgs = df_higgs.rename(columns={'eta': 'h_eta', 'phi': 'h_phi'})

    # Merge Higgs info onto the Selected Fat Jet
    # Note: 'inner' join here will also drop events that somehow don't have a Higgs record
    df_fj_matched = df_fj_selected.merge(df_higgs, left_on='event_index',
                                         right_index=True, how='inner')

    # Calculate dR(Higgs, FatJet)
    df_fj_matched['dr_hfj'] = delta_r(df_fj_matched['eta'],
                                      df_fj_matched['h_eta'],
                                      df_fj_matched['phi'],
                                      df_fj_matched['h_phi'])

    # FILTER: Keep only events where the selected jet is close to the Higgs
    df_fj_final = df_fj_matched[df_fj_matched['dr_hfj'] < 1.0].copy()

    # Get the final list of verified signal events
    final_signal_events = df_fj_final['event_index'].unique()
    print(
        f"  - Events passing Higgs Matching (dR < 1.0): {len(final_signal_events)}")

    # Clean up the fat jet dataframe (drop calc columns)
    df_fj_final = df_fj_final.drop(
        columns=['l_eta', 'l_phi', 'dr_lep', 'h_eta', 'h_phi', 'dr_hfj'])

    # 4. Filter All Other Dataframes to this Final Event List
    df_lep_final = df_lep[
        df_lep['event_index'].isin(final_signal_events)].copy()
    df_j_final = df_j[df_j['event_index'].isin(final_signal_events)].copy()
    df_met_final = df_met[
        df_met['event_index'].isin(final_signal_events)].copy()

    # 5. Filter Constituents
    keys = df_fj_final[['event_index', 'index']]
    df_fjc_final = df_fjc.merge(keys, left_on=['event_index', 'ind'],
                                right_on=['event_index', 'index'], how='inner')
    df_fjc_final = df_fjc_final.loc[:, ~df_fjc_final.columns.duplicated()]

    # 6. Save to Parquet
    print(f"  - Saving {len(df_lep_final)} lepton rows")
    df_lep_final.to_parquet(f"{OUTPUT_DIR}/signal_df_lep.parquet")

    print(f"  - Saving {len(df_j_final)} jet rows")
    df_j_final.to_parquet(f"{OUTPUT_DIR}/signal_df_j.parquet")

    print(f"  - Saving {len(df_fj_final)} fat jet rows (1 per event)")
    df_fj_final.to_parquet(f"{OUTPUT_DIR}/signal_df_fj.parquet")

    print(f"  - Saving {len(df_met_final)} MET rows")
    df_met_final.to_parquet(f"{OUTPUT_DIR}/signal_df_met.parquet")

    print(f"  - Saving {len(df_fjc_final)} constituent rows")
    df_fjc_final.to_parquet(f"{OUTPUT_DIR}/signal_df_fjc.parquet")


filter_and_save_signal(signal_df_lep, signal_df_j, signal_df_fj, signal_df_met, signal_df_fjc, signal_df_g)

backgrounds = [
    ("background_280", background_280_df_lep, background_280_df_j, background_280_df_fj, background_280_df_met, background_280_df_fjc),
    ("background_500", background_500_df_lep, background_500_df_j, background_500_df_fj, background_500_df_met, background_500_df_fjc),
    ("background_1000", background_1000_df_lep, background_1000_df_j, background_1000_df_fj, background_1000_df_met, background_1000_df_fjc),
]

for bg_name, bg_lep, bg_j, bg_fj, bg_met, bg_fjc in backgrounds:
    filter_and_save(bg_name, bg_lep, bg_j, bg_fj, bg_met, bg_fjc)

print("\nAll files filtered and saved to Parquet.")