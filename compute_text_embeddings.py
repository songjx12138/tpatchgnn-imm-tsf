import os
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from fusions.load_llm import (
    load_llm,
    embed_notes,
    get_d_model,
    get_default_max_length,
)


def compute_text_embeddings(
    data_name: str,
    llm_model_fusion: str,
    llm_layers_fusion: int | None,
    max_length: int | None = None,
    device: str = "cpu",
    use_device_map: bool = False,
) -> None:
    """
    Loop over all records in base_dir, read each text.csv, embed notes one at a time,
    and save text_embeddings_{llm_model_fusion}_{llm_layers_fusion or 'full'}.pt

    Args:
      data_name: name of the dataset (e.g. 'ILINet', 'FNSPID')
      llm_model_fusion: key or model ID (e.g. 'GPT2')
      llm_layers_fusion: number of layers to keep, or None for all
      max_length: maximum length of input tokens (None means auto by model)
      device: 'cpu' or 'cuda'
    """
    if max_length is None:
        max_length = get_default_max_length(llm_model_fusion)

    d_model = get_d_model(llm_model_fusion)
    print(f"Using max_length={max_length}, expected embedding dim={d_model}")

    base_dir = f"data/{data_name}/processed"
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    # Load LLM once
    tokenizer, llm_model = load_llm(
        llm_model_fusion,
        llm_layers_fusion,
        device,
        use_device_map=use_device_map,
    )

    # Discover all record subfolders
    record_ids = sorted(
        [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    )
    if not record_ids:
        raise RuntimeError(f"No record subfolders under {base_dir}")

    # Iterate records with progress bar
    for idx, rec in enumerate(record_ids):
        print(f"[{idx + 1}/{len(record_ids)}] Processing record: {rec}")
        rec_dir = os.path.join(base_dir, rec)
        text_csv = os.path.join(rec_dir, "text.csv")
        if not os.path.isfile(text_csv):
            tqdm.write(f"[SKIP] no text.csv in {rec_dir}")
            continue

        # Prepare output path
        out_name = (
            f"text_embeddings_model={llm_model_fusion}"
            f"_layers={llm_layers_fusion or 'full'}"
            f"_maxlen={max_length}.pt"
        )
        out_path = os.path.join(rec_dir, out_name)

        # Skip if output already exists
        if os.path.isfile(out_path):
            tqdm.write(f"[SKIP] Embeddings already exist for '{rec}', skipping.")
            continue

        tqdm.write(f"Embedding notes for record '{rec}'...")
        df = pd.read_csv(text_csv, parse_dates=["date_time"])
        # Use time_series.csv min time as base (consistent with parse_datasets.py)
        ts_csv = os.path.join(rec_dir, "time_series.csv")
        ts_df = pd.read_csv(ts_csv, usecols=["date_time"])
        ts_df["date_time"] = pd.to_datetime(ts_df["date_time"])
        base_ts = ts_df["date_time"].min()
        # Store rel_times in raw seconds (unit conversion happens at load time)
        text_cols = [c for c in df.columns if c not in ("date_time", "record_id")]
        if len(text_cols) != 1:
            raise ValueError(f"{rec_dir}: expected 1 text col, got {text_cols}")
        text_col = text_cols[0]
        # Drop rows with NaN text (consistent with parse_datasets.py raw text path)
        df = df.dropna(subset=[text_col]).reset_index(drop=True)
        rel_times = ((df["date_time"] - base_ts).dt.total_seconds()).tolist()
        notes = df[text_col].astype(str).tolist()

        # Embed each note one by one to save memory
        embeddings = []
        for note in tqdm(notes, desc=f"Notes/{rec}", leave=False, unit="note"):
            emb, _ = embed_notes([[note]], tokenizer, llm_model, max_length=max_length)
            embeddings.append(emb.squeeze(0).squeeze(0).cpu())
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        # Stack into Tensor [N_notes, d_model]
        if embeddings:
            emb_tensor = torch.stack(embeddings, dim=0)
        else:
            emb_tensor = torch.empty((0, d_model), dtype=torch.float32)

        if emb_tensor.size(1) != d_model:
            raise ValueError(
                f"Embedding dim mismatch for {rec}: got {emb_tensor.size(1)}, expected {d_model}"
            )

        # Save embeddings and rel_times
        torch.save(
            {
                "embeddings": emb_tensor,
                "rel_times": torch.tensor(rel_times, dtype=torch.float32),
                "embedding_dim": d_model,
                "llm_model_fusion": llm_model_fusion,
                "llm_layers_fusion": llm_layers_fusion,
                "max_length": max_length,
            },
            out_path,
        )
        tqdm.write(f"Wrote embeddings to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precompute text embeddings for each record's text.csv."
    )
    parser.add_argument(
        "--data_name_list",
        nargs="+",
        default=["MIMIC"],
        help="Dataset names under data/<dataset>/processed",
    )
    parser.add_argument(
        "--llm_model_fusion",
        type=str,
        default="GPT2",
        help="Alias or HuggingFace model id (e.g., ClinicalBERT or emilyalsentzer/Bio_ClinicalBERT).",
    )
    parser.add_argument("--llm_layers_fusion", type=int, default=None)
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Token max length. If unset, uses model-specific default.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--use_device_map",
        action="store_true",
        help="Use transformers device_map='auto' when loading the model.",
    )
    cli_args = parser.parse_args()

    device = cli_args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    max_length = cli_args.max_length or get_default_max_length(cli_args.llm_model_fusion)

    print(f"### LLM model: {cli_args.llm_model_fusion} ###")
    print(f"### max_length={max_length}, device={device} ###")

    for data_name in cli_args.data_name_list:
        print(f"### Processing dataset: {data_name} ###")
        compute_text_embeddings(
            data_name=data_name,
            llm_model_fusion=cli_args.llm_model_fusion,
            llm_layers_fusion=cli_args.llm_layers_fusion,
            max_length=max_length,
            device=device,
            use_device_map=cli_args.use_device_map,
        )
