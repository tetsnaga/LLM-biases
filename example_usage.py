#!/usr/bin/env python3
"""
Simple example script demonstrating all four evidence levels.
"""

import sys
import os

# Add the current directory to Python path so we can import the modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the sampling function
from sample_claims import sample_claims_dataset

def main():
    """
    Run examples for all four evidence levels.
    """
    
    print("Evidence Level Examples")
    print("=" * 30)
    
    # Check if input file exists
    if not os.path.exists("Data/dataset.json"):
        print("❌ Error: dataset.json not found in current directory")
        return
    
    # Evidence Level 1: No evidence
    print("\n1️⃣ Evidence Level 1: No evidence")
    sample_claims_dataset("dataset.json", evidence_level=1)
    
    # Evidence Level 2: Basic evidence
    print("\n2️⃣ Evidence Level 2: Basic evidence")
    sample_claims_dataset("dataset.json", evidence_level=2)
    
    # Evidence Level 3: Evidence with source type
    print("\n3️⃣ Evidence Level 3: Evidence with source type")
    sample_claims_dataset("dataset.json", evidence_level=3)
    
    # Evidence Level 4: Evidence with claim entropy
    print("\n4️⃣ Evidence Level 4: Evidence with claim entropy")
    sample_claims_dataset("dataset.json", evidence_level=4)
    
    print("\n✅ All evidence levels completed!")
    print("\nGenerated files:")
    print("- claims_EL1.json (no evidence)")
    print("- claims_EL2.json (basic evidence)")
    print("- claims_EL3.json (evidence + source type)")
    print("- claims_EL4.json (evidence + claim entropy)")

if __name__ == "__main__":
    main()
