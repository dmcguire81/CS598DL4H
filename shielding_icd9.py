import sys
import pandas as pd

def main(shielding_csv, mapping_csv):
    shielding_df = pd.read_csv(shielding_csv)
    shielding_icd10_df = shielding_df[shielding_df.CodeType == "ICD-10"]
    shielding_icd10_df.set_index("Code", inplace=True)

    mapping_df = pd.read_csv(mapping_csv)
    mapping_df["ICD9"] = mapping_df["Pure Victorian Logical"]
    mapping_df.drop(columns=["Pure Victorian Logical"], inplace=True)
    mapping_df.set_index("ICD10", inplace=True)

    shielding_icd9_df = shielding_icd10_df.join(mapping_df)
    print(shielding_icd9_df.info())
    shielding_icd9_df.to_csv("shielding_icd9.csv", header=True)

if __name__ == "__main__":
    main(*sys.argv[1:])
