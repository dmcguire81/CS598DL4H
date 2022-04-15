import sys

import pandas as pd


def insert_period(icd9_code):
    return f"{icd9_code[:3]}.{icd9_code[3:]}" if len(icd9_code) > 3 else icd9_code


def main(shielding_csv, mapping_csv, d_icd_diagnoses_csv, diagnosis_icd_csv):
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

    unmatched_shiedling_icd9_df = shielding_icd9_df[shielding_icd9_df.ICD9.isnull()]
    print(f"{len(unmatched_shiedling_icd9_df)} unmatched ICD9 codes:")
    print(unmatched_shiedling_icd9_df)

    d_icd_diagnoses_df = pd.read_csv(d_icd_diagnoses_csv)
    d_icd_diagnoses_df.set_index("ICD9_CODE", inplace=True)

    shielding_icd9_df.set_index("ICD9", inplace=True, drop=False)
    shielding_icd9_df.drop(columns=["CodeType", "TABLETYP"], inplace=True)
    shielding_diagnoses_df = shielding_icd9_df.join(d_icd_diagnoses_df)
    shielding_diagnoses_df.reset_index(drop=True, inplace=True)
    matched_shielding_diagnoses_df = shielding_diagnoses_df[
        ~shielding_diagnoses_df.ROW_ID.isnull()
    ]
    print()
    print(f"{len(matched_shielding_diagnoses_df)} matched diagnoses.")
    print(matched_shielding_diagnoses_df.info())
    print(matched_shielding_diagnoses_df)
    # Since this is joined with D_ICD_DIAGNOSES.csv, results can't get checked in
    matched_shielding_diagnoses_df.to_csv("shielding_diagnoses.csv", index=False)
    matched_shielding_diagnoses_df.set_index("ICD9", inplace=True, drop=False)

    diagnosis_icd_df = pd.read_csv(diagnosis_icd_csv)
    diagnosis_icd_df.drop(columns=["ROW_ID"], inplace=True)
    diagnosis_icd_df.set_index("ICD9_CODE", inplace=True)

    shielding_diagnoses_prevalence_df = (
        matched_shielding_diagnoses_df[["ICD9"]]
        .join(diagnosis_icd_df[["HADM_ID"]])
        .groupby(["ICD9"])
        .nunique()
    )
    frequent_shielding_diagnoses_df = shielding_diagnoses_prevalence_df[
        shielding_diagnoses_prevalence_df.HADM_ID > 50
    ]
    frequent_shielding_diagnoses_df.reset_index(inplace=True, drop=False)
    frequent_shielding_diagnoses_df["ICD9"] = frequent_shielding_diagnoses_df[
        "ICD9"
    ].map(insert_period)
    print(frequent_shielding_diagnoses_df[["ICD9"]])
    # Since this is joined with D_ICD_DIAGNOSES.csv and DIAGNOSIS_ICD.csv, results can't get checked in
    # This should go in caml-mimic/mimicdata/mimic3/ along side ALL_CODES.csv and TOP_50_CODES.csv
    frequent_shielding_diagnoses_df[["ICD9"]].to_csv(
        "SHIELDING_CODES.csv", index=False, header=False
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
