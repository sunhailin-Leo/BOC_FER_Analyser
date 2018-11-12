# -*- coding: UTF-8 -*- #
"""
Created on 2018年11月5日
@author: Leo
"""
from pyecharts import Line, Style, Kline


def draw_line_pic(
        title: str,
        labels: list,
        data_package: list,
        x_axis: list,
        y_min: int = 0,
        y_max: int = 100,
        y_formatter: str = '元人民币',
        is_datazoom: bool = False,
        datazoom_range: list = [80, 100],
        path: str = './charts/line.html'
        ):
    """
    折线图
    :param title: 画图标题
    :param labels: 图例
    :param data_package: 数据包
    :param x_axis: x轴的数据
    :param y_min: y轴最小值
    :param y_max: y轴最大值
    :param y_formatter: y轴的formatter
    :param is_datazoom: 是否使用datazoom
    :param datazoom_range: datazoom数据范围
    :param path: 保存的路径
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
                 label_text_size=8,
                 label_text_color="",
                 label_emphasis_textsize=10,
                 xaxis_rotate=-10,
                 xaxis_max="dataMax",
                 yaxis_min=y_min,
                 yaxis_max=y_max,
                 yaxis_formatter=y_formatter,
                 # mark_point=["max", "min"],
                 mark_line=['min', 'max', 'average'],
                 is_datazoom_show=is_datazoom,
                 datazoom_type="both",
                 datazoom_range=datazoom_range
                 )
    line.render(path=path)


def draw_kline_pic(
        title: str,
        labels: list,
        data_package: list,
        y_min: int = 0,
        y_max: int = 100,
        y_formatter: str = "元人民币",
        path: str = './charts/k_line.html'):
    """
    K线图
    :param title: 画图标题
    :param labels: 图例
    :param data_package: 数据包
    :param y_min: y轴最小值
    :param y_max: y轴最大值
    :param y_formatter: y轴的格式化
    :param path: 保存的路径
    """
    style = Style(
        title_top="#fff",
        title_pos="left",
        width=1920,
        height=900
    )
    kline = Kline(title=title, renderer='svg', **style.init_style)
    kline.add('日K', labels, data_package,
              yaxis_min=y_min,
              yaxis_max=y_max,
              yaxis_formatter=y_formatter,
              mark_line=["min", "max"],
              mark_point=["min", "max"],
              is_datazoom_show=True,
              datazoom_type="both",
              datazoom_range=[80, 100])
    kline.render(path=path)
