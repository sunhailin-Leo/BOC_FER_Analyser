# -*- coding: UTF-8 -*- #
"""
Created on 2018年11月11日
@author: Leo
"""

from utils.data_loader import DataLoader


def load_dataframe():
    data_conn = DataLoader(password='123456')
    df = data_conn.load_dataframe(db_name='exchange_rate',
                                  table_name="t_exchange_rate_hkd",
                                  currency_name="港币")
    print(df)
    return df


if __name__ == '__main__':
    load_dataframe()
