import pandas as pd


def xlsx_to_csv_pd():
    data_xls = pd.read_excel('C:/CCF/TF-IDF+Kmeans/data/微贷网.xlsx', index_col=0)
    data_xls.to_csv('C:/CCF/TF-IDF+Kmeans/data/微贷网.csv', encoding='utf_8_sig')


if __name__ == '__main__':
    xlsx_to_csv_pd()