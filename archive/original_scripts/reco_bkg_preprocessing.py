import uproot
import awkward as ak
import pyarrow as pa
import pyarrow.parquet as pq

bkg_file = uproot.open("data_reco_raw/sig_noMuConst.root")
bkg_tree = bkg_file["analysis"]

# 1. Define the exact fj_track columns we want to keep
allowed_fj_tracks = set()
for i in range(1, 31):
    allowed_fj_tracks.update([
        f"fj_track{i}Pt",
        f"fj_track{i}Eta",
        f"fj_track{i}Phi",
    ])

# 2. Filter the tree's keys before loading
all_keys = bkg_tree.keys()
keys_to_keep = []
for key in all_keys:
    if not key.startswith("fj_track"):
        keys_to_keep.append(key)
    elif key in allowed_fj_tracks:
        keys_to_keep.append(key)

# --- NEW CHUNKED PROCESSING LOGIC ---

output_file = "df_sig_noMu.parquet"
writer = None

# 3. Iterate through the file in chunks (e.g., 500 MB at a time)
for chunk in bkg_tree.iterate(keys_to_keep, step_size="500 MB"):

    # 4. Create the boolean mask for the current chunk
    mask1 = ak.any(chunk["muon_fatjet_dr"] < 1.2, axis=1)
    mask2 = ak.any(chunk["fatJetM"] > 20, axis=1)
    combined_mask = mask1 & mask2

    filtered_chunk = chunk[combined_mask]

    # 5. Convert the filtered chunk to a Pandas DataFrame
    df_chunk = ak.to_dataframe(filtered_chunk)

    # 6. Append to the Parquet file incrementally
    if not df_chunk.empty:
        # Convert pandas chunk to an Arrow table
        table = pa.Table.from_pandas(df_chunk)

        # Initialize the Parquet writer on the first non-empty chunk
        if writer is None:
            writer = pq.ParquetWriter(output_file, table.schema)

        # Write the chunk to the file
        writer.write_table(table)

# 7. Close the writer when the loop finishes
if writer is not None:
    writer.close()
    print(f"Successfully processed and saved to {output_file}")
else:
    print("No events passed the filter conditions.")