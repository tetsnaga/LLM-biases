"""
Sample script for filtered climate claims dataset (JSON)
Randomly samples 90 SUPPORTS and 90 REFUTES claims
"""

import pandas as pd
import sys
import os
import random
import json

def filter_evidences(df, evidence_level=2):
    """
    Filter evidences array based on evidence level.
    
    Args:
        df (DataFrame): DataFrame with evidences column
        evidence_level (int): Evidence level (2 or 3)
        
    Returns:
        DataFrame: DataFrame with filtered evidences
    """
    def filter_single_evidence(evidences):
        """Filter a single evidence array"""
        if not isinstance(evidences, list):
            return evidences
        
        filtered_evidences = []
        for evidence in evidences:
            if isinstance(evidence, dict):
                if evidence_level == 2:
                    # Keep only evidence_id and evidence fields
                    filtered_evidence = {
                        'evidence_id': evidence.get('evidence_id', ''),
                        'evidence': evidence.get('evidence', '')
                    }
                elif evidence_level == 3:
                    # Keep only evidence_id and evidence (source_type will be added at claim level)
                    filtered_evidence = {
                        'evidence_id': evidence.get('evidence_id', ''),
                        'evidence': evidence.get('evidence', '')
                    }
                elif evidence_level == 4:
                    # Keep only evidence_id and evidence (claim_entropy will be added at claim level)
                    filtered_evidence = {
                        'evidence_id': evidence.get('evidence_id', ''),
                        'evidence': evidence.get('evidence', '')
                    }
                else:
                    # Default to level 2 behavior
                    filtered_evidence = {
                        'evidence_id': evidence.get('evidence_id', ''),
                        'evidence': evidence.get('evidence', '')
                    }
                
                filtered_evidences.append(filtered_evidence)
            else:
                # If not a dict, keep as is
                filtered_evidences.append(evidence)
        
        return filtered_evidences
    
    # Apply filtering to evidences column
    df_copy = df.copy()
    if 'evidences' in df_copy.columns:
        df_copy['evidences'] = df_copy['evidences'].apply(filter_single_evidence)
    
    return df_copy

def sample_claims_dataset(input_file, output_file=None, sample_size=30, random_seed=42, evidence_level=2):
    """
    Sample the claims dataset by randomly selecting from each stance_label, claim_label combination.
    
    Args:
        input_file (str): Path to the input JSON file (sorted dataset)
        output_file (str): Path to the output JSON file (optional)
        sample_size (int): Number of rows to sample per stance_label, claim_label combination (default: 30)
        random_seed (int): Random seed for reproducibility (default: 42)
        evidence_level (int): Evidence exposure level (default: 2)
                             1 = No evidence (evidences column removed)
                             2 = Evidence included (only evidence_id and evidence fields)
                             3 = Evidence included (evidence_id, evidence, and source_type fields)
                             4 = Evidence included (evidence_id, evidence, and claim_entropy fields)
    """
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return False
    
    try:
        # Set random seed for reproducibility
        random.seed(random_seed)
        
        # Read the JSON file
        print(f"Reading claims dataset from {input_file}...")
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format in {input_file}: {str(e)}")
            return False
        except Exception as e:
            print(f"Error reading JSON file: {str(e)}")
            return False
        
        # Convert JSON to DataFrame
        df = pd.DataFrame(data)
        
        print(f"Original dataset shape: {df.shape}")
        
        # Check if required columns exist
        required_columns = ['stance_label', 'claim_label']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            return False
        
        # Get unique combinations of stance_label and claim_label
        unique_combinations = df[['stance_label', 'claim_label']].drop_duplicates()
        print(f"\nFound {len(unique_combinations)} unique stance_label, claim_label combinations:")
        for _, row in unique_combinations.iterrows():
            count = len(df[(df['stance_label'] == row['stance_label']) & (df['claim_label'] == row['claim_label'])])
            print(f"  {row['stance_label']} + {row['claim_label']}: {count} rows")
        
        # Sample rows for each combination
        sampled_rows = []
        
        print(f"\nSampling {sample_size} rows for each stance_label, claim_label combination...")
        
        for _, combo in unique_combinations.iterrows():
            stance = combo['stance_label']
            claim = combo['claim_label']
            
            # Get all rows with this combination
            combo_rows = df[(df['stance_label'] == stance) & (df['claim_label'] == claim)]
            available_count = len(combo_rows)
            
            print(f"  {stance} + {claim}: {available_count} available rows")
            
            if available_count >= sample_size:
                # Sample sample_size rows
                sampled = combo_rows.sample(n=sample_size, random_state=random_seed)
                print(f"    -> Sampled {sample_size} rows")
            else:
                # If not enough rows available, take all available rows
                sampled = combo_rows
                print(f"    -> Only {available_count} rows available, taking all")
            
            sampled_rows.append(sampled)
        
        # Combine all sampled rows
        df_sampled = pd.concat(sampled_rows, ignore_index=True)
        
        # Shuffle the final dataset to mix the stance_label values
        df_sampled = df_sampled.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        
        # Handle evidence level
        print(f"\nEvidence level: {evidence_level}")
        if evidence_level == 1:
            # Remove evidences column if it exists
            if 'evidences' in df_sampled.columns:
                df_sampled = df_sampled.drop('evidences', axis=1)
                print("  -> Evidences column removed (evidence_level = 1)")
            else:
                print("  -> Evidences column not found in dataset")
        elif evidence_level == 2:
            # Filter evidences to show only evidence_id and evidence
            if 'evidences' in df_sampled.columns:
                print("  -> Filtering evidences to show only evidence_id and evidence (evidence_level = 2)")
                df_sampled = filter_evidences(df_sampled, evidence_level=2)
            else:
                print("  -> Evidences column not found in dataset")
        elif evidence_level == 3:
            # Filter evidences and add source_type at claim level based on claim_label
            if 'evidences' in df_sampled.columns:
                print("  -> Filtering evidences and adding source_type at claim level (evidence_level = 3)")
                df_sampled = filter_evidences(df_sampled, evidence_level=3)
                
                # Add source_type field to each claim based on claim_label
                def get_source_type(claim_label):
                    if claim_label in ['SUPPORTS', 'REFUTES']:
                        return 'reputable source'
                    else:  # NOT_ENOUGH_INFO or any other value
                        return 'unknown source'
                
                df_sampled['source_type'] = df_sampled['claim_label'].apply(get_source_type)
            else:
                print("  -> Evidences column not found in dataset")
        elif evidence_level == 4:
            # Filter evidences and keep existing claim_entropy field
            if 'evidences' in df_sampled.columns:
                print("  -> Filtering evidences and keeping claim_entropy field (evidence_level = 4)")
                df_sampled = filter_evidences(df_sampled, evidence_level=4)
            else:
                print("  -> Evidences column not found in dataset")
        else:
            print(f"  -> Warning: Invalid evidence_level {evidence_level}. Using default (2)")
            evidence_level = 2
        
        # Remove rationale and confidence columns for all evidence levels
        columns_to_remove = ['rationale', 'confidence']
        
        # Remove claim_entropy for all evidence levels except 4
        if evidence_level != 4:
            columns_to_remove.append('claim_entropy')
        
        # Remove source_type for all evidence levels except 3
        if evidence_level != 3:
            columns_to_remove.append('source_type')
        
        for col in columns_to_remove:
            if col in df_sampled.columns:
                df_sampled = df_sampled.drop(col, axis=1)
                print(f"  -> Removed {col} column")
        
        print(f"\nFinal dataset shape after evidence processing: {df_sampled.shape}")
        print(f"\nFinal distribution:")
        final_counts = df_sampled['stance_label'].value_counts()
        for value, count in final_counts.items():
            print(f"  {value}: {count} rows")
        
        print(f"\nFinal distribution by stance_label, claim_label combinations:")
        combo_counts = df_sampled.groupby(['stance_label', 'claim_label']).size()
        for (stance, claim), count in combo_counts.items():
            print(f"  {stance} + {claim}: {count} rows")
        
        # Generate output filename if not provided
        if output_file is None:
            output_file = f"claims_EL{evidence_level}.json"
        
        # Save the sampled dataset
        print(f"\nSaving sampled dataset to {output_file}...")
        
        # Convert DataFrame to JSON format
        sampled_data = df_sampled.to_dict('records')
        
        # Save as JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sampled_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nSampling complete!")
        print(f"Output file: {output_file}")
        print(f"Total sampled rows: {len(df_sampled)}")
        
        # Show first few rows
        print(f"\nFirst 5 rows of sampled dataset:")
        for idx, row in df_sampled.head(5).iterrows():
            claim_preview = row['claim'][:60] + "..." if len(str(row['claim'])) > 60 else row['claim']
            print(f"  Row {idx + 1}: {row['stance_label']} | {claim_preview}")
        
        # Show sample of each stance_label, claim_label combination
        print(f"\nSample claims by stance_label, claim_label combinations:")
        for stance in df_sampled['stance_label'].unique():
            stance_data = df_sampled[df_sampled['stance_label'] == stance]
            print(f"\n{stance} stance_label:")
            for claim in stance_data['claim_label'].unique():
                combo_data = stance_data[stance_data['claim_label'] == claim]
                print(f"  {claim} claim_label:")
                for idx, row in combo_data.head(2).iterrows():
                    claim_preview = row['claim'][:80] + "..." if len(str(row['claim'])) > 80 else row['claim']
                    print(f"    - {claim_preview}")
        
        return True
        
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")
        return False

def main():
    """Main function to handle command line arguments"""
    
    # Default input file
    default_input = "Data/climate-fever-dataset.json"
    
    # Parse command line arguments
    input_file = default_input
    output_file = None
    sample_size = 30
    random_seed = 42
    evidence_level = 2
    
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
        elif args[i] == "--evidence-level" or args[i] == "-e":
            if i + 1 < len(args):
                try:
                    evidence_level = int(args[i + 1])
                    if evidence_level not in [1, 2, 3, 4]:
                        print("Error: --evidence-level must be 1, 2, 3, or 4")
                        sys.exit(1)
                    i += 2
                except ValueError:
                    print("Error: --evidence-level must be an integer (1, 2, 3, or 4)")
                    sys.exit(1)
            else:
                print("Error: --evidence-level requires a number (1, 2, 3, or 4)")
                sys.exit(1)
        elif args[i] == "--help" or args[i] == "-h":
            print("Usage: python sample_claims.py [options]")
            print("Options:")
            print("  -i, --input FILE        Input JSON file (default: climate-fever-dataset.json)")
            print("  -o, --output FILE      Output JSON file (default: claims_EL{level}.json)")
            print("  -s, --sample-size N    Number of rows to sample per stance_label, claim_label combination (default: 30)")
            print("  -r, --seed N           Random seed for reproducibility (default: 42)")
            print("  -e, --evidence-level N Evidence exposure level (default: 2)")
            print("                        1 = No evidence (evidences column removed)")
            print("                        2 = Evidence included (only evidence_id and evidence fields)")
            print("                        3 = Evidence included (evidence_id, evidence, and source_type fields)")
            print("                        4 = Evidence included (evidence_id, evidence, and claim_entropy fields)")
            print("  -h, --help             Show this help message")
            sys.exit(0)
        else:
            # If no flag, treat as positional argument
            if input_file == default_input:
                input_file = args[i]
            elif output_file is None:
                output_file = args[i]
            i += 1
    
    print("Climate Claims Dataset Sampler (JSON)")
    print("=" * 50)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file or f'claims_EL{evidence_level}.json'}")
    print(f"Sample size per stance_label: {sample_size}")
    print(f"Random seed: {random_seed}")
    print(f"Evidence level: {evidence_level}")
    print()
    
    # Run the sampling
    success = sample_claims_dataset(input_file, output_file, sample_size, random_seed, evidence_level)
    
    if success:
        print("\nSampling completed successfully!")
    else:
        print("\nSampling failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
