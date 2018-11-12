# -*- coding: UTF-8 -*- #
"""
Created on 2018年11月5日
@author: Leo
"""
# Python第三方库
import numpy as np
import pandas as pd
import pyflux as pf
import statsmodels.api as sm
from matplotlib import pyplot as plt

# 项目内部库
from utils.draw_pic import *
from utils.decorators import check_path
from utils.data_loader import DataLoader


def get_dataframe(data_conn) -> pd.DataFrame:
    """
    获取Dataframe
    :param data_conn: 数据连接对象
    :return:
    """
    df = data_conn.load_dataframe(db_name='exchange_rate',
                                  table_name="t_exchange_rate",
                                  currency_name="港币")
    return df


def custom_dataframe_handler(df: pd.DataFrame) -> pd.DataFrame:
    """
    自定义dataframe处理
    :param df: Dataframe
    """
    df = df.sort_values(by='查询时间')

    # 转换数据类型
    df['现汇买入价'] = df['现汇买入价'].astype('float')
    df['现钞买入价'] = df['现钞买入价'].astype('float')
    df['现汇卖出价'] = df['现汇卖出价'].astype('float')
    df['现钞卖出价'] = df['现钞卖出价'].astype('float')
    df['中行折算价'] = df['中行折算价'].astype('float')

    # 去重
    df = df.drop_duplicates(subset='查询时间', keep='first')

    return df


@check_path(path='./charts/')
def draw_pic(dataframe: pd.DataFrame, charts_type: str = 'kline'):
    """
    画图(使用pyecharts对自定义后的数据进行画图, 并不是建模后的数据)
    :param dataframe: Dataframe
    :param charts_type: 画图类型 默认为kline (均值K线)
    """

    def _kline(df: pd.DataFrame, columns_name: str = '现汇卖出价', title: str = '人民币和港币的汇率折算(100港币)'):
        # K线图数据
        df['查询时间'] = df['查询时间'].apply(lambda x: x[:-9])
        df['查询时间'] = pd.to_datetime(df['查询时间'], format="%Y-%m-%d")
        df = df.groupby('查询时间')[columns_name]
        labels = []
        values = []
        for d in df:
            temp_data = d[1].tolist()
            k_data = [temp_data[0], temp_data[-1], min(temp_data), max(temp_data)]
            labels.append(str(d[0])[:-9])
            values.append(k_data)
        draw_kline_pic(title=title, labels=labels, data_package=values)

    def _line(df: pd.DataFrame, title: str = '人民币和港币的汇率折算(100港币)'):
        header = ['现汇买入价', '现钞买入价', '现汇卖出价', '现钞卖出价']
        # 折线图数据
        total_data = [
            df['现汇买入价'].tolist(), df['现钞买入价'].tolist(), df['现汇卖出价'].tolist(), df['现钞卖出价'].tolist()
        ]
        draw_line_pic(
            title=title,
            labels=header,
            data_package=total_data,
            x_axis=df['查询时间'].tolist()
        )

    # 判断使用那个画图方式
    if charts_type == 'kline':
        _kline(df=dataframe)
    elif charts_type == 'line':
        _line(df=dataframe)
    else:
        raise ValueError('画图类型参数不正确!')


def first_difference_restore(
        origin_df: pd.DataFrame,
        predict_df: pd.DataFrame,
        columns_name: str) -> pd.DataFrame:
    """
    一阶查分还原(二阶或者高阶的需要自行编写)
    :param origin_df: 原始数据的Dataframe
    :param predict_df: 预测结果的Dataframe
    :param columns_name: 目标列名(暂时只支持一列)
    :return: 返回还原结果
    """
    last_data = origin_df[columns_name].values[-1]
    predict_data_list = predict_df.values.tolist()
    restore_list = []
    for d in predict_data_list:
        last_data = last_data + d
        restore_list.append(last_data)
    predict_data = pd.DataFrame(restore_list, index=predict_df.index, columns=[columns_name])
    return predict_data


@check_path(path='./picture/')
def build_statsmodels_model(
        df: pd.DataFrame,
        diff_num: int = 1,
        model_p: int = 2,
        model_q: int = 2,
        lag_num: int = 40,
        predict_start_time: str = '2018-11-09',
        predict_end_time: str = '2018-11-14',
        model_start_time: str = '2018-01-01',
        model_end_time: str = '',
        restore_columns_name: str = '现汇卖出价') -> pd.DataFrame:
    """
    statsmodels建模
    :param df: Dataframe
    :param diff_num: 差分阶数 默认为1
    :param model_p: ARIMA p参数 默认为2
    :param model_q: ARIMA q参数 默认为2
    :param lag_num: 滞后数 默认为40
    :param predict_start_time: 预测起始时间 默认为 2018-11-09
    :param predict_end_time: 预测结束时间 默认为 2018-11-14
    :param model_start_time: 预测需要的数据起始时间 默认为 2018-01-01
    :param model_end_time: 预测需要的数据结束时间 默认为空
    :param restore_columns_name: 差分还原的列名(暂时支持一列数据) 默认为 现汇卖出价
    :return 返回预测结果
    """
    # 差分图
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(111)
    # 里面的1代表了差分阶数
    diff1 = df.diff(diff_num)
    diff1.plot(ax=ax1)
    fig.show()
    try:
        fig.savefig('./picture/diff_{}.jpg'.format(str(diff_num)))
    except FileNotFoundError:
        pass

    # 自相关图 偏相关图
    diff1 = diff1.dropna()
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(diff1, lags=lag_num, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(diff1, lags=lag_num, ax=ax2)
    fig.show()
    try:
        fig.savefig('./picture/acf_pacf.jpg')
    except FileNotFoundError:
        pass

    # 建模
    # p = 2, q = 2
    arma_mod = sm.tsa.ARMA(diff1, (model_p, model_q)).fit()
    # -195.11076756537022 -172.6720100922948 -186.1416899854131
    print(arma_mod.aic, arma_mod.bic, arma_mod.hqic)

    # 残差(输出形式为DataFrame) 并画出ACF和PACF图
    resid = arma_mod.resid
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=lag_num, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(resid, lags=lag_num, ax=ax2)
    fig.show()
    try:
        fig.savefig('./picture/resid_acf_pacf.jpg')
    except FileNotFoundError:
        pass

    # 残差D-W检验
    resid_dw_result = sm.stats.durbin_watson(arma_mod.resid.values)
    # 1.9933441709003574 接近于 2 所以残差序列不存在自相关性。
    print(resid_dw_result)

    # 正态校验 -> 基本符合正态分布
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    fig = sm.qqplot(resid, line='q', ax=ax, fit=True)
    fig.show()
    try:
        fig.savefig('./picture/normal_distribution.jpg')
    except FileNotFoundError:
        pass

    # 残差序列Ljung-Box检验，也叫Q检验
    r, q, p = sm.tsa.acf(resid.values.squeeze(), qstat=True)
    data = np.c_[range(1, lag_num + 1), r[1:], q, p]
    table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
    temp_df = table.set_index('lag')
    print(temp_df)
    # Prob(>Q)的最小值: 0.025734615668093132, 最大值: 0.9874705305611844, 均值: 0.2782013984159408
    prob_q_min = temp_df['Prob(>Q)'].min()
    prob_q_max = temp_df['Prob(>Q)'].max()
    prob_q_mean = temp_df['Prob(>Q)'].mean()
    print("Prob(>Q)的最小值: {}, 最大值: {}, 均值: {}".format(prob_q_min, prob_q_max, prob_q_mean))

    # 预测
    predict_data = arma_mod.predict(predict_start_time, predict_end_time, dynamic=True)
    # 画预测图
    fig, ax = plt.subplots(figsize=(12, 8))
    if model_end_time == "":
        ax = diff1.ix[model_start_time:].plot(ax=ax)
    else:
        ax = diff1.ix[model_start_time: model_end_time].plot(ax=ax)
    fig = arma_mod.plot_predict(predict_start_time, predict_end_time, dynamic=True, ax=ax, plot_insample=False)
    fig.show()
    try:
        fig.savefig('./picture/predict_pic.jpg')
    except FileNotFoundError:
        pass

    # 结果预测
    predict_data = first_difference_restore(origin_df=df, predict_df=predict_data, columns_name=restore_columns_name)
    print(predict_data)

    # 返回预测结果(可以保存, 需要自行编写保存的方法)
    return predict_data


def build_pyflux_model(
        df: pd.DataFrame,
        model_p: int = 1,
        model_q: int = 1,
        model_diff: int = 0,
        model_predict_len: int = 5,
        model_use_past_values: int = 250,
        model_target_columns: str = '现汇卖出价'):
    """
    pyflux建模
    :param df: Dataframe
    :param model_p: 模型P参数
    :param model_q: 模型q参数
    :param model_diff: 模型差分阶数
    :param model_predict_len: 模型预测时间长度 默认为5 即 5天后的数据
    :param model_use_past_values: 模型使用过去多少天的数据进行预测 默认为250
    :param model_target_columns: 建模的目标列名
    """
    # ARIMA 差分阶数设为了0, pyflux的差分有点问题
    model = pf.ARIMA(
        data=df,
        ar=model_p,
        ma=model_q,
        integ=model_diff,
        target=model_target_columns,
        family=pf.Normal())
    x = model.fit("MLE")
    x.summary()
    # 画图
    model.plot_z(figsize=(15, 10))
    model.plot_fit(figsize=(15, 10))
    model.plot_predict_is(h=model_use_past_values, figsize=(15, 10))
    model.plot_predict(h=model_predict_len, past_values=model_use_past_values, figsize=(15, 10))
    predict_data = model.predict(h=model_predict_len)

    # 差分还原
    if model_diff > 0 and model_diff == 1:
        predict_data = first_difference_restore(
            origin_df=df,
            predict_df=predict_data,
            columns_name=model_target_columns)
    print(predict_data)

    # 返回预测结果(可以保存, 需要自行编写保存的方法)
    return predict_data


def build_model(df: pd.DataFrame, model_class_name: str = 'statsmodels'):
    """
    建模
    :param df: Dataframe
    :param model_class_name: 模型名称
    """
    # 对时间进行格式化求日均值(这一块与画图部分有冲突, 需要自定义的修改这一块就好)
    df['查询时间'] = df['查询时间'].apply(lambda x: x[:-9])
    df['查询时间'] = pd.to_datetime(df['查询时间'], format="%Y-%m-%d")
    df = df.groupby('查询时间')['现汇卖出价'].mean()
    df = df.to_frame()

    if model_class_name == "statsmodels":
        build_statsmodels_model(df=df)
    elif model_class_name == "pyflux":
        build_pyflux_model(df=df)
    else:
        raise ValueError("错误的模型类别名称!")


def main_process():
    # 加载数据
    data = DataLoader()
    data_frame = get_dataframe(data_conn=data)

    # 数据自定义过滤清洗
    data_frame = custom_dataframe_handler(df=data_frame)

    # 画图并保存
    # draw_pic(dataframe=data_frame, charts_type='kline')
    # draw_pic(dataframe=data_frame, charts_type='line')

    # 建模
    # build_model(df=data_frame, model_class_name='statsmodels')
    # build_model(data_frame, model_class_name='pyflux')


if __name__ == '__main__':
    main_process()
