"""
Filter script for climate-fever-dataset.csv
Filters rows to only include stance_label = SUPPORTS or REFUTES
Sorts with SUPPORTS rows first, then REFUTES rows
"""

import pandas as pd
import sys
import os

def filter_dataset(input_file, output_file=None):
    """
    Filter the dataset to only include SUPPORTS and REFUTES stance labels,
    with SUPPORTS rows first and REFUTES rows second.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to the output CSV file (optional)
    """
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return False
    
    try:
        # Read the CSV file
        print(f"Reading dataset from {input_file}...")
        df = pd.read_csv(input_file)
        
        print(f"Original dataset shape: {df.shape}")
        print(f"Original stance_label distribution:")
        print(df['stance_label'].value_counts())
        
        # Filter to only include SUPPORTS and REFUTES
        filtered_df = df[df['stance_label'].isin(['SUPPORTS', 'REFUTES'])]
        
        print(f"\nAfter filtering to SUPPORTS and REFUTES:")
        print(f"Filtered dataset shape: {filtered_df.shape}")
        print(f"Filtered stance_label distribution:")
        print(filtered_df['stance_label'].value_counts())
        
        # Sort with SUPPORTS first, then REFUTES
        # Create a custom sort order
        stance_order = {'SUPPORTS': 0, 'REFUTES': 1}
        filtered_df['stance_order'] = filtered_df['stance_label'].map(stance_order)
        filtered_df = filtered_df.sort_values('stance_order').drop('stance_order', axis=1)
        
        # Reset index
        filtered_df = filtered_df.reset_index(drop=True)
        
        # Generate output filename if not provided
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_filtered.csv"
        
        # Save the filtered dataset
        print(f"\nSaving filtered dataset to {output_file}...")
        filtered_df.to_csv(output_file, index=False)
        
        print(f"\nFiltering complete!")
        print(f"Output file: {output_file}")
        print(f"Final dataset shape: {filtered_df.shape}")
        
        # Show first few rows of each stance
        print(f"\nFirst 3 SUPPORTS rows:")
        supports_rows = filtered_df[filtered_df['stance_label'] == 'SUPPORTS'].head(3)
        for idx, row in supports_rows.iterrows():
            print(f"  Row {idx + 1}: {row['stance_label']} | {row['claim'][:80]}...")
        
        print(f"\nFirst 3 REFUTES rows:")
        refutes_rows = filtered_df[filtered_df['stance_label'] == 'REFUTES'].head(3)
        for idx, row in refutes_rows.iterrows():
            print(f"  Row {idx + 1}: {row['stance_label']} | {row['claim'][:80]}...")
        
        return True
        
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")
        return False

def main():
    """Main function to handle command line arguments"""
    
    # Default input file
    default_input = "climate-fever-dataset.csv"
    
    # Check command line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = default_input
    
    # Check for output file argument
    output_file = None
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    print("Climate Fever Dataset Filter")
    print("=" * 50)
    print(f"Input file: {input_file}")
    if output_file:
        print(f"Output file: {output_file}")
    print()
    
    # Run the filtering
    success = filter_dataset(input_file, output_file)
    
    if success:
        print("\nFiltering completed successfully!")
    else:
        print("\nFiltering failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
