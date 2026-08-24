import pandas as pd
import awkward as ak
import uproot
from joblib import Parallel, delayed
import os
import numpy as np


def flatten_df(original_df, prefix):
    """
    Extracts columns with a given prefix, converts them to a tidy DataFrame,
    and adds an event_index. (This function remains unchanged).
    """
    cols = [c for c in original_df.columns if c.startswith(prefix)]
    cols = [c for c in cols if
            c not in ['fj_tau1', 'fj_tau2', 'fj_tau3', 'fj_tau4', 'fj_N2',
                      'fj_M2']]
    if not cols:
        return None
    ak_dict = {c.replace(prefix, ''): original_df[c] for c in cols}
    ak_records = ak.Array(ak_dict)
    if len(ak_records) == 0:
        return pd.DataFrame()
    flat_df = ak.to_dataframe(ak_records)
    if flat_df.empty:
        return pd.DataFrame()

    if isinstance(flat_df.index, pd.MultiIndex):
        # This is for jagged data (jets, leptons, etc.)
        flat_df = flat_df.reset_index(level=1, drop=True)
        flat_df = flat_df.reset_index().rename(
            columns={'entry': 'event_index'})
    else:
        # This is for scalar data (MET, event weights, etc.)
        # The index is already the event number. Just rename it.
        flat_df = flat_df.reset_index().rename(
            columns={'entry': 'event_index'})
    if prefix != 'fjc_':
        try:
            # Attempt to apply the offset
            flat_df['index'] = flat_df.groupby("event_index").cumcount()
        except KeyError:
            # 1. Catch the specific error
            # 2. Create a much more helpful error message
            error_message = (
                f"CRITICAL ERROR: Failed to find or create the 'event_index' column for prefix='{prefix}'. "
                f"This indicates an unexpected data structure that the function could not handle. "
                f"Printing the dataframe here: {flat_df.columns}."
            )
            # 3. Raise a new exception with the improved message
            raise KeyError(error_message)
    return flat_df


# --- Configuration ---
# Adjust this based on your RAM. It's the number of events to read at once.
STEP_SIZE = 100000

# IMPORTANT: Define the paths to your SOURCE .root files.
SOURCE_DATA = {
    "signal_df": "data_raw/signal.root",
    # "background_280_df": "data_raw/background_280_500.root",
    # "background_500_df": "data_raw/background_500_1000.root",
    # "background_1000_df": "data_raw/background_1000.root"
}
TREENAME = "analysis"  # The name of the TTree inside your .root files

OUTPUT_PATH = "data"
ALL_PREFIXES = ['g_',
    'j_', 'lep_', 'fj_', 'fjc_', 'met_']


# --- Main Processing Function for a Single File and Prefix ---
def process_prefix_for_file(df_name, source_file_path, prefix, output_path):
    """
    Reads a large .root file in chunks using uproot.iterate, processes each chunk
    for a single prefix, and saves the final concatenated result.
    """
    print(f"Starting processing for {df_name} with prefix {prefix}...")

    # Use uproot.iterate to read the .root file in memory-efficient chunks
    # This is the key change from the previous version.
    file_and_tree = f"{source_file_path}:{TREENAME}"
    chunk_iterator = uproot.iterate(file_and_tree, library="pd",
                                    step_size=STEP_SIZE)

    processed_chunks = []
    for chunk in chunk_iterator:
        df_new = flatten_df(chunk, prefix)

        if not df_new.empty:
            # Get the first index value of the current chunk (its absolute starting point)
            chunk_start_index = chunk.index[0]
            # Add this offset to the local event_index to make it globally consistent
            df_new['event_index'] = df_new['event_index'] + chunk_start_index

            processed_chunks.append(df_new)

    if not processed_chunks:
        print(f"No data produced for {df_name} with prefix {prefix}.")
        return

    # Concatenate all processed chunks into a single DataFrame
    final_df = pd.concat(processed_chunks, ignore_index=True)

    # Save the final result
    output_name = df_name + '_' + prefix.strip('_')
    file_name = f"{output_name}.csv"
    final_df.to_csv(os.path.join(output_path, file_name), index=False)
    print(f"✅ Saved {file_name}")


# --- Parallel Execution ---
tasks = []
for df_name, source_path in SOURCE_DATA.items():
    for prefix in ALL_PREFIXES:
        tasks.append((df_name, source_path, prefix, OUTPUT_PATH))

print("🚀 Starting parallel processing...")
Parallel(n_jobs=-1)(delayed(process_prefix_for_file)(*task) for task in tasks)
print("🎉 All tasks completed!")
