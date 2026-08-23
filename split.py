#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd


def split_dataset(path, seed, test_size=0.2):
	df = pd.read_csv(path, header=None)

	df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

	split_idx = int(len(df) * (1 - test_size))

	train_df = df.iloc[:split_idx]
	val_df = df.iloc[split_idx:]

	return train_df, val_df


def main():
	parser = argparse.ArgumentParser(description="Split CSV into train/val (80/20)")
	parser.add_argument("--input", required=True, help="Input CSV file path")
	parser.add_argument("--test_size", type=float, default=0.2, help="Validation split ratio")

	args = parser.parse_args()

	seed = np.random.randint(0, 2**32 - 1)

	train_df, val_df = split_dataset(args.input, seed, args.test_size)

	train_path = "data/processed/train.csv"
	val_path = "data/processed/val.csv"

	train_df.to_csv(train_path, index=False, header=False)
	val_df.to_csv(val_path, index=False, header=False)

	print(f"Saved train set to {train_path}")
	print(f"Saved val set to {val_path}")


if __name__ == "__main__":
	main()