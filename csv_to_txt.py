import pandas as pd
import sys


def main(caml_mimic_csv_file, hlan_text_file):
    df = pd.read_csv(caml_mimic_csv_file)
    print(df.info())
    
    text_lines = df["TEXT"].tolist()
    label_lines = df["LABELS"].map(lambda line: line.replace(";", " ")).tolist()
    
    with open(hlan_text_file, "w") as fh:
        for text_line, label_line in zip(text_lines, label_lines):
            print(f"{text_line}__label__{label_line}", file=fh)
        

if __name__ == "__main__":
    main(*sys.argv[1:])
