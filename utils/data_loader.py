# -*- coding: UTF-8 -*- #
"""
Created on 2018年11月5日
@author: Leo
"""

# Python第三方库
import pymysql
import pandas as pd

# 项目内部库
from utils.enum_variable import SELECT_SQL


class DataLoader:
    def __init__(self,
                 host: str ='localhost',
                 port: int = 3306,
                 user: str = 'root',
                 password: str = ''):
        # MySQL配置项
        self._host = host
        self._port = port
        self._user = user
        self._password = password

        # 获取MySQL连接
        self._conn = self._sql_conn()

        # Dataframe配置
        # 表头
        self.header = \
            ['现汇买入价', '现钞买入价', '现汇卖出价', '现钞卖出价', '中行折算价', '查询时间']

        # dataframe配置
        # 显示所有列
        pd.set_option('display.max_columns', None)
        # 显示所有行
        # pd.set_option('display.max_rows', None)
        # 设置value的显示长度为100，默认为50
        pd.set_option('max_colwidth', 100)

    def _sql_conn(self) -> pymysql.connect:
        """
        连接MySQL
        """
        try:
            conn = pymysql.connect(host=self._host,
                                   port=self._port,
                                   user=self._user,
                                   password=self._password)
            return conn
        except Exception as err:
            print(err)

    def load_dataframe(self,
                       db_name: str,
                       table_name: str,
                       currency_name: str) -> pd.DataFrame:
        """
        把数据加载到Dataframe中
        :param db_name: 数据库名称
        :param table_name: 表名称
        :param currency_name: 条件名称
        :return Dataframe
        """
        # 拼接SQL
        execute_sql = SELECT_SQL.format(db_name, table_name, currency_name)

        # 加载Dataframe
        df = pd.read_sql(sql=execute_sql, con=self._conn)
        df.columns = self.header
        return df
