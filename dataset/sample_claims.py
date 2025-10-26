#!/usr/bin/env python3
"""
Sample script for filtered climate claims dataset
Randomly samples 90 SUPPORTS and 90 REFUTES claims
"""

import pandas as pd
import sys
import os
import random

def sample_claims_dataset(input_file, output_file=None, sample_size=90, random_seed=42):
    """
    Sample the claims dataset by randomly selecting SUPPORTS and REFUTES claims.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to the output CSV file (optional)
        sample_size (int): Number of rows to sample per stance_label (default: 90)
        random_seed (int): Random seed for reproducibility (default: 42)
    """
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return False
    
    try:
        # Set random seed for reproducibility
        random.seed(random_seed)
        
        # Read the CSV file
        print(f"Reading claims dataset from {input_file}...")
        df = pd.read_csv(input_file)
        
        print(f"Original dataset shape: {df.shape}")
        
        # Check if stance_label column exists
        if 'stance_label' not in df.columns:
            print("Error: stance_label column not found in the dataset.")
            print(f"Available columns: {list(df.columns)}")
            return False
        
        # Get unique values of stance_label
        unique_values = df['stance_label'].unique()
        print(f"\nFound stance_label values: {unique_values}")
        
        # Show original distribution
        print(f"\nOriginal distribution:")
        original_counts = df['stance_label'].value_counts()
        for value, count in original_counts.items():
            print(f"  {value}: {count} rows")
        
        # Check if we have SUPPORTS and REFUTES
        required_labels = ['SUPPORTS', 'REFUTES']
        missing_labels = [label for label in required_labels if label not in unique_values]
        
        if missing_labels:
            print(f"Error: Missing required stance_label values: {missing_labels}")
            print(f"Available values: {unique_values}")
            return False
        
        # Sample rows for SUPPORTS and REFUTES
        sampled_rows = []
        
        print(f"\nSampling {sample_size} rows for each stance_label...")
        
        for label in required_labels:
            # Get all rows with this stance_label
            label_rows = df[df['stance_label'] == label]
            available_count = len(label_rows)
            
            print(f"  {label}: {available_count} available rows")
            
            if available_count >= sample_size:
                # Sample sample_size rows
                sampled = label_rows.sample(n=sample_size, random_state=random_seed)
                print(f"    -> Sampled {sample_size} rows")
            else:
                # If not enough rows available, take all available rows
                sampled = label_rows
                print(f"    -> Only {available_count} rows available, taking all")
            
            sampled_rows.append(sampled)
        
        # Combine all sampled rows
        df_sampled = pd.concat(sampled_rows, ignore_index=True)
        
        # Shuffle the final dataset to mix the stance_label values
        df_sampled = df_sampled.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        
        print(f"\nSampled dataset shape: {df_sampled.shape}")
        
        # Show final distribution
        print(f"\nFinal distribution:")
        final_counts = df_sampled['stance_label'].value_counts()
        for value, count in final_counts.items():
            print(f"  {value}: {count} rows")
        
        # Generate output filename if not provided
        if output_file is None:
            output_file = "sampled_claims.csv"
        
        # Save the sampled dataset
        print(f"\nSaving sampled dataset to {output_file}...")
        df_sampled.to_csv(output_file, index=False)
        
        print(f"\nSampling complete!")
        print(f"Output file: {output_file}")
        print(f"Total sampled rows: {len(df_sampled)}")
        
        # Show first few rows
        print(f"\nFirst 5 rows of sampled dataset:")
        for idx, row in df_sampled.head(5).iterrows():
            claim_preview = row['claim'][:60] + "..." if len(str(row['claim'])) > 60 else row['claim']
            print(f"  Row {idx + 1}: {row['stance_label']} | {claim_preview}")
        
        # Show sample of each stance
        print(f"\nSample claims by stance:")
        for stance in ['SUPPORTS', 'REFUTES']:
            stance_rows = df_sampled[df_sampled['stance_label'] == stance].head(2)
            print(f"\n{stance} examples:")
            for idx, row in stance_rows.iterrows():
                claim_preview = row['claim'][:80] + "..." if len(str(row['claim'])) > 80 else row['claim']
                print(f"  - {claim_preview}")
        
        return True
        
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")
        return False

def main():
    """Main function to handle command line arguments"""
    
    # Default input file
    default_input = "processing/climate-fever-dataset_filtered.csv"
    
    # Parse command line arguments
    input_file = default_input
    output_file = None
    sample_size = 90
    random_seed = 42
    
    # Simple argument parsing
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--input" or args[i] == "-i":
            if i + 1 < len(args):
                input_file = args[i + 1]
                i += 2
            else:
                print("Error: --input requires a filename")
                sys.exit(1)
        elif args[i] == "--output" or args[i] == "-o":
            if i + 1 < len(args):
                output_file = args[i + 1]
                i += 2
            else:
                print("Error: --output requires a filename")
                sys.exit(1)
        elif args[i] == "--sample-size" or args[i] == "-s":
            if i + 1 < len(args):
                try:
                    sample_size = int(args[i + 1])
                    i += 2
                except ValueError:
                    print("Error: --sample-size must be an integer")
                    sys.exit(1)
            else:
                print("Error: --sample-size requires a number")
                sys.exit(1)
        elif args[i] == "--seed" or args[i] == "-r":
            if i + 1 < len(args):
                try:
                    random_seed = int(args[i + 1])
                    i += 2
                except ValueError:
                    print("Error: --seed must be an integer")
                    sys.exit(1)
            else:
                print("Error: --seed requires a number")
                sys.exit(1)
        elif args[i] == "--help" or args[i] == "-h":
            print("Usage: python sample_claims.py [options]")
            print("Options:")
            print("  -i, --input FILE        Input CSV file (default: processing/climate-fever-dataset_filtered.csv)")
            print("  -o, --output FILE      Output CSV file (default: sampled_claims.csv)")
            print("  -s, --sample-size N    Number of rows to sample per stance_label (default: 90)")
            print("  -r, --seed N           Random seed for reproducibility (default: 42)")
            print("  -h, --help             Show this help message")
            sys.exit(0)
        else:
            # If no flag, treat as positional argument
            if input_file == default_input:
                input_file = args[i]
            elif output_file is None:
                output_file = args[i]
            i += 1
    
    print("Climate Claims Dataset Sampler")
    print("=" * 50)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file or 'sampled_claims.csv'}")
    print(f"Sample size per stance_label: {sample_size}")
    print(f"Random seed: {random_seed}")
    print()
    
    # Run the sampling
    success = sample_claims_dataset(input_file, output_file, sample_size, random_seed)
    
    if success:
        print("\nSampling completed successfully!")
    else:
        print("\nSampling failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
