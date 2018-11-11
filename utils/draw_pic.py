# -*- coding: UTF-8 -*- #
"""
Created on 2018年11月5日
@author: Leo
"""
from pyecharts import Line, Style, Kline


def draw_line_pic(title: str, labels: list, data_package: list, x_axis: list):
    """
    折线图
    :param title:
    :param labels:
    :param data_package:
    :param x_axis:
    :return:
    """
    style = Style(
        title_top="#fff",
        title_pos="left",
        width=1920,
        height=900
    )

    line = Line(title=title, renderer='svg', **style.init_style)
    for i, d in enumerate(labels):
        line.add(d, x_axis, data_package[i],
                 is_stack=False,
                 is_label_show=True,
                 is_smooth=True,
                 legend_selectedmode='single',
                 yaxis_min=78,
                 yaxis_max=90,
                 yaxis_formatter="元人民币",
                 # mark_point=["max", "min"],
                 mark_line=['min', 'max', 'average'],
                 is_datazoom_show=True,
                 datazoom_type="both",
                 datazoom_range=[80, 100]
                 )
    line.render(path='./charts/line.html')


def draw_kline_pic(title: str, labels: list, data_package: list):
    """
    K线图
    :param title:
    :param labels:
    :param data_package:
    :return:
    """
    style = Style(
        title_top="#fff",
        title_pos="left",
        width=1920,
        height=900
    )
    kline = Kline(title=title, renderer='svg', **style.init_style)
    kline.add('日K', labels, data_package,
              yaxis_min=78,
              yaxis_max=90,
              yaxis_formatter="元人民币",
              mark_line=["min", "max"],
              mark_point=["min", "max"],
              is_datazoom_show=True,
              datazoom_type="both",
              datazoom_range=[80, 100])
    kline.render('./charts/k_line.html')
