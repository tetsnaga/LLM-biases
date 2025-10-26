#!/usr/bin/env python3
"""
Process script for synthetic_climate_personas.csv
Removes Belief_HumanContribution column and sorts by Belief_ClimateExists
"""

import pandas as pd
import sys
import os

def process_personas_dataset(input_file, output_file=None):
    """
    Process the personas dataset by removing Belief_HumanContribution column
    and sorting by Belief_ClimateExists.
    
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
        print(f"Reading personas dataset from {input_file}...")
        df = pd.read_csv(input_file)
        
        print(f"Original dataset shape: {df.shape}")
        print(f"Original columns: {list(df.columns)}")
        
        # Check if Belief_HumanContribution column exists
        if 'Belief_HumanContribution' not in df.columns:
            print("Warning: Belief_HumanContribution column not found in the dataset.")
            print(f"Available columns: {list(df.columns)}")
            return False
        
        # Remove the Belief_HumanContribution column
        df_processed = df.drop('Belief_HumanContribution', axis=1)
        
        print(f"\nAfter removing Belief_HumanContribution column:")
        print(f"Processed dataset shape: {df_processed.shape}")
        print(f"Remaining columns: {list(df_processed.columns)}")
        
        # Check Belief_ClimateExists values and create sort order
        print(f"\nBelief_ClimateExists unique values:")
        unique_values = df_processed['Belief_ClimateExists'].unique()
        print(f"Found values: {unique_values}")
        
        # Create custom sort order for Belief_ClimateExists
        # Assuming the order should be from most disagree to most agree
        climate_order = {
            'Strongly disagree': 0,
            'Slightly disagree': 1,
            'Neutral': 2,
            'Slightly agree': 3,
            'Strongly agree': 4
        }
        
        # Add sort order column
        df_processed['climate_order'] = df_processed['Belief_ClimateExists'].map(climate_order)
        
        # Sort by the climate order
        df_processed = df_processed.sort_values('climate_order').drop('climate_order', axis=1)
        
        # Reset index
        df_processed = df_processed.reset_index(drop=True)
        
        # Generate output filename if not provided
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_processed.csv"
        
        # Save the processed dataset
        print(f"\nSaving processed dataset to {output_file}...")
        df_processed.to_csv(output_file, index=False)
        
        print(f"\nProcessing complete!")
        print(f"Output file: {output_file}")
        print(f"Final dataset shape: {df_processed.shape}")
        
        # Show distribution of Belief_ClimateExists after sorting
        print(f"\nBelief_ClimateExists distribution after sorting:")
        climate_counts = df_processed['Belief_ClimateExists'].value_counts()
        for value, count in climate_counts.items():
            print(f"  {value}: {count} rows")
        
        # Show first few rows of each Belief_ClimateExists category
        print(f"\nFirst 3 rows for each Belief_ClimateExists category:")
        for value in climate_counts.index:
            rows = df_processed[df_processed['Belief_ClimateExists'] == value].head(3)
            print(f"\n{value}:")
            for idx, row in rows.iterrows():
                print(f"  Row {idx + 1}: {row['PersonaID']} | {row['AgeGroup']} | {row['Gender']} | {row['Region']}")
        
        return True
        
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")
        return False

def main():
    """Main function to handle command line arguments"""
    
    # Default input file
    default_input = "synthetic_climate_personas.csv"
    
    # Check command line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = default_input
    
    # Check for output file argument
    output_file = None
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    print("Synthetic Climate Personas Dataset Processor")
    print("=" * 60)
    print(f"Input file: {input_file}")
    if output_file:
        print(f"Output file: {output_file}")
    print()
    
    # Run the processing
    success = process_personas_dataset(input_file, output_file)
    
    if success:
        print("\nProcessing completed successfully!")
    else:
        print("\nProcessing failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
