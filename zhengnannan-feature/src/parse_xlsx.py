# -*- coding:utf-8 -*-

import argparse
import pandas as pd
from openpyxl import load_workbook


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_file', help='path to xlsx file')
    parser.add_argument('output_file', help='path to output file')
    args = parser.parse_args()

    wb = load_workbook(filename=args.input_file)
    sheets = wb.get_sheet_names()
    ws = wb.get_sheet_by_name(sheets[0])
    rows = ws.rows
    columns = ws.columns

    content = []
    for row in rows:
        line = [col.value for col in row]
        content.append(line)

    df = pd.DataFrame(content)
    df.to_csv(args.output_file, sep='\t', index=False, header=False, encoding='utf-8')

