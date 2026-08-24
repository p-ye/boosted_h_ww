import pandas as pd
import awkward as ak
import uproot
from joblib import Parallel, delayed
import os
import numpy as np

# --- Configuration ---
# Adjust this based on your RAM. It's the number of events to read at once.
STEP_SIZE = 100000

# IMPORTANT: Define the paths to your SOURCE .root files.
SOURCE_DATA = {
    "signal_df": "data_reco_raw/sig.root",

    # "background_280_df": "data_raw/background_280_500.root",
    # "background_500_df": "data_raw/background_500_1000.root",
    # "background_1000_df": "data_raw/background_1000.root"
}
TREENAME = "analysis"  # The name of the TTree inside your .root files

OUTPUT_PATH = "data_reco"
ALL_PREFIXES = ['g_',
                'j_', 'lep_', 'fj_', 'fjc_', 'met_']


def process_prefix_for_file(df_name, source_file_path, prefix, output_path):
    """
    Processes data using an "Awkward Native" approach for maximum performance.
    """
    print(f"Starting processing for {df_name} with prefix {prefix}...")

    output_name = df_name + '_' + prefix.strip('_')
    file_path = os.path.join(output_path, f"{output_name}.csv")

    # Get a list of all branches that match the prefix
    with uproot.open(f"{source_file_path}:{TREENAME}") as tree:
        all_branches = tree.keys()
        prefix_branches = [b for b in all_branches if b.startswith(prefix)]
        prefix_branches = [b for b in prefix_branches if
                           b not in ['fj_tau1', 'fj_tau2', 'fj_tau3',
                                     'fj_tau4', 'fj_N2',
                                     'fj_M2']]

    if not prefix_branches:
        print(
            f"No branches found for prefix '{prefix}' in {df_name}. Skipping.")
        return

    # Use library="ak" to get Awkward Arrays directly
    chunk_iterator = uproot.iterate(
        f"{source_file_path}:{TREENAME}",
        expressions=prefix_branches,  # Only read the columns we need
        library="ak",
        step_size=STEP_SIZE
    )

    is_first_chunk = True
    entry_counter = 0
    for chunk in chunk_iterator:
        # 1. Rename fields (branches) by creating a new record
        new_fields = {b.replace(prefix, ''): chunk[b] for b in prefix_branches}
        processed_chunk = ak.zip(new_fields)

        if len(processed_chunk) == 0:
            continue

        # ✅ FIX STEP 1: Calculate and add the event_index BEFORE filtering.
        start_event_index = entry_counter

        # We add the index to the original, unfiltered chunk.
        if processed_chunk.ndim == 1:  # Scalar data
            event_index = ak.local_index(processed_chunk,
                                         axis=0) + start_event_index
        else:  # Jagged data
            first_field = processed_chunk[processed_chunk.fields[0]]
            event_index = \
            ak.broadcast_arrays(ak.local_index(processed_chunk, axis=0),
                                first_field)[0]+start_event_index

        # Add the index as a new field. Now all fields have the same length.
        processed_chunk["event_index"] = event_index

        # ✅ FIX STEP 3: The event_index is now jagged for jagged data. Flatten it.
        # This is the final step before converting to pandas.
        if processed_chunk.ndim > 1:
            processed_chunk = ak.flatten(processed_chunk, axis=1)

        df_new = ak.to_dataframe(processed_chunk)

        if not df_new.empty:
            if is_first_chunk:
                df_new.to_csv(file_path, index=False, mode='w', header=True)
                is_first_chunk = False
            else:
                df_new.to_csv(file_path, index=False, mode='a', header=False)

        entry_counter += len(chunk)
    if is_first_chunk:
        print(f"No data produced for {df_name} with prefix {prefix}.")
    else:
        print(f"✅ Saved {output_name}.csv. entry counter: {entry_counter}.")


# --- Parallel Execution ---
tasks = []
for df_name, source_path in SOURCE_DATA.items():
    for prefix in ALL_PREFIXES:
        tasks.append((df_name, source_path, prefix, OUTPUT_PATH))

print("🚀 Starting parallel processing...")
Parallel(n_jobs=-1)(delayed(process_prefix_for_file)(*task) for task in tasks)
print("🎉 All tasks completed!")
