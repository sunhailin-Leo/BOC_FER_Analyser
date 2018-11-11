# -*- coding: UTF-8 -*- #
"""
Created on 2018年11月5日
@author: Leo
"""

SELECT_SQL = "SELECT buying_rate, cash_buying_rate, selling_rate, cash_selling_rate, boe_conversion_rate, rate_time " \
             "FROM " + "{}.{}" + " WHERE currency_name = \'{}\'"
