## 中国银行外汇牌价分析 (Analyse from Bank Of China exchange rate data)

---

<h3 id="Developer">开发者</h3>

* Core: sunhailin-Leo
* E-mail: 379978424@qq.com
* Wechat: 18666270636

* **在使用中如果有啥功能需求或者出现BUG的话, 欢迎提ISSUE或者直接联系我~**

---

<h3 id="DevEnv">开发环境</h3>

* 系统环境: Windows 10 x64
* Python版本: Python 3.6
* 编译器: Pycharm

---

<h3 id="ProjectInfo">项目简介</h3>

* 项目简介:
    1. 分析从中国银行外汇牌价爬虫抓取下来的数据(数据可视化和数据预测)
        * 前提是数据已存入到数据库中
        
    2. 分别使用pyflux和statsmodels对数据进行预测分析
        * statsmodels默认将建模中的图存储到项目根目录下的picture文件夹中
        * pyflux由于源码中不支持保存图片, 因此暂时无法保存到本地
    
    3. 使用pyecharts进行数据可视化和结果图导出

* 启动简介:
    
```html
# 暂时自行修改代码使用不同类型的模型
python analyser.py
```

---

<h3 id="OthersMention">其他注意事项</h3>

1. 数据可视化方面将渲染的引擎默认使用了svg如果需要用canvas的自行在utils/draw_pic中修改
    * 数据量大的情况下不建议使用折线图
    
2. 建模选择pyflux的时候画出的图无法保存到本地(源码画图的时候写死了plt.show()) --> 比较汗颜

3. 修改代码使用statsmodels可以选择加入check_path这个装饰器(加入装饰器会自动创建文件夹, 将建模过程中使用的图进行保存, 默认开启)

---

<h3 id="Future">未来开发方向</h3>

* 尝试使用机器学习或者深度学习的方法进行建模预测(可能会在别的分支或者新建一个仓库进行)

---

<h3 id="ChangeLog">更新文档</h3>

* 版本 v1.0 - 2018-11-11:
    * 双十一不剁手就写代码嘿嘿~
    * 提交数据挖掘的源代码

* 版本 v1.1 - 2018-11-12:
    * 更新draw_pic的代码, 部分参数引入到函数参数中
    * 修改analyser中的部分代码和参数引用