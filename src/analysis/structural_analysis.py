import pandas as pd
import json
from pathlib import Path
import os
from src.protocol import results_ledger_path

# Dataset Taxonomy Mapping
DOMAIN_MAP = {
    "bank-marketing": "Finance",
    "credit-g": "Finance",
    "default-of-credit-card-clients": "Finance",
    "breast-cancer-wisconsin": "Healthcare",
    "heart-disease": "Healthcare",
    "diabetes": "Healthcare",
    "haberman": "Healthcare",
    "adult": "Demographics",
    "aps_failure": "Industry",
    "electricity": "Tech/Utility",
    "PhishingWebsites": "Cybersecurity",
    "jm1": "Software Engineering",
    "covertype": "Agriculture/Nature",
    "dry-bean-dataset": "Agriculture/Nature",
    "mushroom": "Agriculture/Nature",
    "magic-telescope": "Astronomy",
    "ionosphere": "Tech/Physics",
    "sonar": "Tech/Physics",
    "spambase": "Tech/Cyber",
    "wine-quality-red": "Industry/Food"
}

def load_data():
    # Load Results
    records = []
    results_path = str(results_ledger_path())
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except:
                        pass
    df_results = pd.DataFrame(records)

    # Load Meta-features
    meta_records = []
    for meta_file in Path("data/raw").glob("*_meta.json"):
        dataset_name = meta_file.name.replace("_meta.json", "")
        with open(meta_file, "r") as f:
            meta = json.load(f)
            meta["dataset"] = dataset_name
            meta["domain"] = DOMAIN_MAP.get(dataset_name, "Other")
            meta_records.append(meta)
    df_meta = pd.DataFrame(meta_records)
    
    return df_results, df_meta

def main():
    raise RuntimeError('Unpaired legacy structural association and its figures are retired for corrected runs.')

if __name__ == '__main__':
    main()

