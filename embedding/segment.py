# -*- coding:utf-8 -*-

import argparse
import utils
from openpyxl import load_workbook


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_file', help='path to xlsx file')
    parser.add_argument('segment_file', help='path to segment file')
    args = parser.parse_args()
    raw_input = args.input_file
    segmentfilePath = args.segment_file
    wb = load_workbook(filename=raw_input)
    sheets = wb.get_sheet_names()
    ws = wb.get_sheet_by_name(sheets[0])
    rows = ws.rows

    segmentfile = open(segmentfilePath, 'w')
    count = 0
    for row in rows:
        line = [col.value for col in row]
        tc = utils.TextCleaner()
        seg = utils.Segmentor()
        text = line[1].encode('utf-8')
        # print text, type(text)
        text = tc.clean(text)
        seg_text = seg.seg_token(text)
        if seg_text:
            segmentfile.write(' '.join(seg_text).encode('utf-8') + '\n')  # ' ' for word2vec by gensim
        if count % 1000 == 0:
            print 'have processed %d lines.' % (count)
        count += 1
    segmentfile.flush()
    segmentfile.close()
