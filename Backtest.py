# -*- coding: utf-8 -*-
"""
Created on Wed May 29 14:13:53 2019

@author: Hilbert
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import time
import warnings
warnings.filterwarnings("ignore")


"""
Back test Mode
"""

class Back_test():

    """
    初始化模块
    """
    def __init__(self, start_day, end_day, pick_start, pick_end, selling_limit = 1.09, rise = 1.09, fall = 0.91, rise_st = 1.04, stamp_tax = 0.001, trade_tax = 0.00015):
        self.start_day = start_day
        self.end_day = end_day
        self.pick_start = pick_start #第一只股票
        self.pick_end = pick_end #最后一只股票
        self.selling_limit = selling_limit #卖出阈值,当股票价格涨幅超过(selling_limit - 1)时卖出
        self.rise = rise #高开阈值,当个股高开幅度大于(rise -1),则设为无法买入
        self.fall = fall #低开阈值,当个股低开幅度大于(fall -1),则不买
        self.rise_st = rise_st #ST股票高开阈值
        self.stamp_tax = stamp_tax #印花税, 一般为0.001, 单边仅卖出
        self.trade_tax = trade_tax #佣金, 一般为0.0003, 双边
        self.N = 3513 #股票个数

    """
    载入数据模块
    """
    def Data_load(self):
        self.close = np.load('data/close.npy') #收盘价
        self.open = np.load('data/open.npy') #开盘价
        self.high = np.load('data/high.npy') #最高价
        self.turn = np.load('data/turn.npy') #换手率
        self.st = np.load('data/ST.npy') #ST列表, -1 : 未上市或退市, 1 : ST, 0 : 正常
        self.zz500_close = np.load('data/ZZ500_close.npy') #中证500收盘价
        self.amt = np.load('data/amt.npy') #成交金额
#       self.zz500_open = np.load('data/ZZ500_open.npy') #中证500开盘价
        self.stocklist = np.load('data/stocklist.npy') #股票列表
        self.timeline = np.load('data/timeline.npy') #时间轴
#       self.win_stock = np.load('data/win_stock.npy') #个股每日胜率
        """以上数据计算可计算出如下数据"""
        self.start_index = np.where(self.timeline == self.start_day)[0][0] #时间轴开始索引
        self.end_index = np.where(self.timeline == self.end_day)[0][0] #时间轴结束索引
        self.M = self.end_index - self.start_index + 1 #时间跨度

    """
    剔除模块
    1.明日turn = 0 (停牌或退市)
    2.今明无开收数据 (系统错误)
    3.明开 / 今收 > rise (无法买入)
    4.ST,若考虑买ST则剔除 明开 / 今收 > rise_st (无法买入)
    """
    def Select_stock_array(self, grade, grade_start, ST = False):

        """
        grade无序, array格式, 包括每只股票的得分, M*N
        M为日期跨度, 从start_day开始
        N为股票个数, 顺序与self.stocklist对应
        默认不买ST
        """

        grade_start_index = self.start_index - np.where(self.timeline == grade_start)[0][0] 
        grade_end_index = grade_start_index + self.M - 1
        grade = grade[grade_start_index : grade_end_index + 1]
        grade[np.isnan(grade)] = 0
#       amt_median = np.median(self.amt, axis = 1)

        """处理grade数据,将无法买入的设置为0"""

        for T in range(self.M): #在第T天, 对应时间轴上的索引为 今天 : start_index + T, 明天 : start_index + T + 1
            today = self.start_index + T
            tomorrow = self.start_index + T + 1
            turnIndex = np.argwhere(self.turn[tomorrow] == 0)
            openIndexToday = np.argwhere(np.isnan(self.open[today]))
            openIndexTomorrow = np.argwhere(np.isnan(self.open[tomorrow]))
            closeIndexToday = np.argwhere(np.isnan(self.close[today]))
            closeIndexTomorrow = np.argwhere(np.isnan(self.close[tomorrow]))
            riseUpIndex = np.argwhere(self.open[tomorrow] / self.close[today] > self.rise)
            limitDownIndex = np.argwhere(self.open[tomorrow] / self.close[today] < self.fall)
            if not ST:
                stIndex = np.argwhere(self.st[today] == 1)
                grade[T, stIndex] = 0
            grade[T, turnIndex] = 0
            grade[T, openIndexToday] = 0
            grade[T, openIndexTomorrow] = 0
            grade[T, closeIndexToday] = 0
            grade[T, closeIndexTomorrow] = 0
            grade[T, riseUpIndex] = 0
            grade[T, limitDownIndex] = 0


        grade[grade == 0] = grade.min() - 1
        """创建array存储每日可买股票列表"""
        
        stockindex_daily = np.argsort(- grade, axis = 1)[:, self.pick_start - 1 : self.pick_end]
        return stockindex_daily

#   def Select_stock_excel(self, stock_excel, ST = False):
#
#       """
#       stock_excel为DataFrame, 包括每日购买的股票代码, 相应得分从小到大
#       M为日期跨度, 从start_day开始
#       默认不买ST
#       """
#
#       """构建一个虚拟的grade矩阵"""
#
#       stock_excel_column_length = stock_excel.shape[1] #给定excel表长度, 必须 stock_excel_column_length > self.pick_start - self.pick_end + 1
#       grade = np.zeros((self.M, self.N)) #构建grade
#       f = lambda x: np.argwhere(self.stocklist == x)[0][0] #x为股票代码, 函数返回股票代码在stocklist中的索引
#       for i in range(self.M):
#           stock_daily = stock_excel.iloc[i, :] #取stock_excel第i行
#           m = map(f,stock_daily) #f在stock_daily上的映射
#           picklist = np.array(list(m)) #选取股票的索引
#           grade[i, picklist] = np.arange(1, stock_excel_column_length + 1) 
#
#       """处理grade数据,将无法买入的设置为0"""
#
#       for T in range(self.M): #在第T天, 对应时间轴上的索引为 今天 : start_index + T, 明天 : start_index + T + 1
#           today = self.start_index + T
#           tomorrow = self.start_index + T + 1
#           for S in range(self.N): #S为股票索引,以下4个判断对应上述4个选择
#               if self.turn[tomorrow, S] == 0:
#                   grade[T, S] = 0
#               elif np.isnan(self.open[today, S]) or np.isnan(self.open[tomorrow, S]) or np.isnan(self.close[today, S]) or np.isnan(self.close[tomorrow, S]):
#                   grade[T, S] = 0
#               elif self.open[tomorrow, S] / self.close[today, S] > self.rise:
#                   grade[T, S] = 0
#               elif self.st[today, S] == 1: #判断是否为ST
#                   if ST: #ST = True 代表买ST
#                       if self.open[tomorrow, S] / self.close[today, S] > self.rise_st: #无法买入ST
#                           grade[T, S] = 0
#                   else: #不买ST
#                       grade[T, S] = 0
#               else:
#                   pass
#
#       """创建array存储每日可买股票列表"""
#
#       stockindex_daily = np.argsort(- grade, axis = 1)[:, self.pick_start - 1 : self.pick_end] 
#       return stockindex_daily

    """
    回测模块
    剔除模块将近似涨停无法买入的股票除去，但未除去第二天跌停无法卖出的情况
    对于第二天跌停无法卖出的股票, 当作第二天买入的股票，买入价格为今日开盘价
    维护矩阵matrix_stock, M * (N + 2)
    M为日期跨度
    N为股票个数
    第一列为净值 = 现金 + 股票价值
    第二列为现金
    后N列为个股持仓情况
    """

    def Back_test_mode(self, stockindex_daily):

        """
        策略:
        if cash > stock_value 购股金额 = (cash + sum(stock_value)) / 2
        else 购股金额 = cash 
        """
        
        """每日跌停列表"""
#       limitdown_stock = pd.DataFrame(index = self.timeline[self.start_index : self.end_index + 1], columns = range(self.pick_end - self.pick_start + 1))
        limitdown_stock = []

        """第一天T = 0,以每日收盘结算,最后一个交易日(self.end_day)不买入"""

        cash = 1 #现金
        stock_value = 0 #股票价值
        equity = 1 #净值
        current_index = 0
        matrix_stock = np.zeros((self.M, self.N + 2))
        matrix_stock[0, 0] = equity
        matrix_stock[0, 1] = cash

        """第二天T = 1"""
        
        today = self.start_index + 1
        cash_buystock = 0.5 # cash_buystock = (cash + stock_value) / 2
        cash = cash - cash_buystock #剩余现金
        stock_index = stockindex_daily[0, :] #购买股票索引(array)
        current_index = stock_index #未跌停持仓索引(array,今日买入的)
        stock_value = (cash_buystock * (1 - self.trade_tax) / current_index.shape[0]) * (self.close[today, current_index] / self.open[today, current_index]) #持仓价值(array)
        equity = cash + sum(stock_value) #净值
        matrix_stock[1, 0] = equity
        matrix_stock[1, 1] = cash
        matrix_stock[1, current_index + 2] = stock_value #更新仓位市值

        """第三天到第M天"""

        """考虑卖出股票的情况，如果无法卖出则返回股票索引，进入持仓"""

        def Sell_stock(current_index, limitdown_length, stock_value, T): #这里前limitdown_length的股票是昨天跌停的，后面的是昨天买入今天要卖的
            yesterday = self.start_index + T - 1
            today = self.start_index + T
            limitdown_index = []
            limitdown_value = []
            cash_sellstock = 0
            
            """此为昨天跌停股票, 允许今日开盘价卖出, 无法卖出继续进入跌停仓位"""
            
            for i in range(limitdown_length):
                index = current_index[i]
                rate = self.open[today, index] / self.close[yesterday, index]
                if self.turn[today, index] == 0: #今日停牌直接进入跌停列表
                    limitdown_index.append(index)
                    limitdown_value.append(self.close[today, index] / self.close[yesterday, index] * stock_value[i])
                elif round(round(self.close[yesterday, index], 2) * 0.9 + 0.00001, 2) >= round(self.open[today, index], 2) and self.open[today, index] == self.close[today, index] and self.open[today, index] == self.high[today, index]: #今日开盘跌停且收盘跌停，则继续进入跌停仓位
                    limitdown_index.append(index)
                    limitdown_value.append(self.close[today, index] / self.close[yesterday, index] * stock_value[i])
                else:
                    cash_sellstock += rate * stock_value[i] * (1 - self.trade_tax - self.stamp_tax) #否则以当日开盘价卖出, 成交额减去佣金和印花税
                    
            """此为昨天买入今天欲卖股票, 以今日收盘价卖出, 无法卖出进入跌停仓位"""
            
            for i in range(limitdown_length, len(current_index)):
                index = current_index[i]
                rate = self.close[today, index] / self.close[yesterday, index]
                if self.turn[today, index] == 0: #今日停牌
                    limitdown_index.append(index)
                    limitdown_value.append(self.close[today, index] / self.close[yesterday, index] * stock_value[i])
                elif round(round(self.close[yesterday, index], 2) * 0.9, 2) >= round(self.close[today, index], 2):
                    limitdown_index.append(index)
                    limitdown_value.append(rate * stock_value[i])
                else:
                    cash_sellstock += rate * stock_value[i] * (1 - self.trade_tax - self.stamp_tax) #以收盘价卖出,成交额减去佣金和印花税
            limitdown_index = np.array(limitdown_index, dtype = 'int64')
            limitdown_value = np.array(limitdown_value)
            return cash_sellstock, limitdown_index, limitdown_value #卖出股票所得现金, 跌停股票索引, 跌停股票市值
        
        limitdown_index = [] #跌停票索引
        limitdown_length = len(limitdown_index)

        for T in range(2, self.M):
            
#           yesterday = self.start_index + T - 1
            today = self.start_index + T #today在timeline上的索引
            
            """购买金额"""
            
            if cash > sum(stock_value):
                cash_buystock = (cash + sum(stock_value)) / 2
            else:
                cash_buystock = cash
            stock_index = stockindex_daily[T - 1, :] #购买股票索引
#           print('=============')
#           print(limitdown_index,limitdown_length,current_index, self.timeline[today],T)
            cash_sellstock, limitdown_index, limitdown_value = Sell_stock(current_index, limitdown_length, stock_value, T) #卖出现金, 跌停股票索引, 跌停股票市值
#           print(limitdown_index, self.stocklist[limitdown_index])
#           print('=============')
            limitdown_length = len(limitdown_index)
            limitdown_stock.append([self.timeline[today], self.stocklist[limitdown_index]])

            stock_todayvalue = (cash_buystock * (1 - self.trade_tax) / stock_index.shape[0]) * (self.close[today, stock_index] / self.open[today, stock_index])
            if sum([x == y for x in limitdown_index for y in stock_index]) == 0: #跌停股票不在买入列表中
                stock_value = np.append(limitdown_value, stock_todayvalue) #当前持仓价值(array)
                current_index = np.append(limitdown_index, stock_index) #当前持仓索引
            else:
                stock_value = np.append(limitdown_value, stock_todayvalue) #当前持仓价值(array)
                current_index = np.append(limitdown_index, stock_index) #当前持仓索引
                m = len(limitdown_index)
                ind = []
                for i in range(len(limitdown_index)):
                    for j in range(len(stock_index)):
                        if limitdown_index[i] == stock_index[j]:
                            ind.append(m + j)
                            stock_value[i] += stock_value[m + j]
                current_index = np.delete(current_index, ind)
                stock_value = np.delete(stock_value, ind)   
            cash = cash - cash_buystock + cash_sellstock #现金
            equity = cash + sum(stock_value)
            matrix_stock[T, 0] = equity
            matrix_stock[T, 1] = cash
            matrix_stock[T, current_index + 2] = stock_value

        return matrix_stock, limitdown_stock
    
    """
    存储模块
    """
    
    def Save_data(self, matrix_stock, limitdown_stock, name = 'Alpha'):
        stock_value = matrix_stock[:, 2:] #持仓价值
        df_position = pd.DataFrame(index = self.timeline[self.start_index : self.end_index + 1], columns = range((self.pick_end - self.pick_start + 1) * 2 + 2))
#       df_test = pd.DataFrame(index = self.timeline[self.start_index + 1 : self.end_index + 1], columns = range(20))
#       df_limitdown = pd.DataFrame(index = self.timeline[self.start_index : self.end_index + 1], columns = range(self.N))
        df_position.iloc[0,:2] = np.array([1,1])
        maxlen = 0
        for i in range(1, self.M):
            stock_index = np.where(stock_value[i] != 0)[0]
            stock_code = self.stocklist[stock_index]
            m = len(stock_code)
            maxlen = max(maxlen, m)
            df_position.iloc[i, 0] = matrix_stock[i, 0]
            df_position.iloc[i, 1] = matrix_stock[i, 1]
            df_position.iloc[i, 2 : m + 2] = stock_code
        df_position = df_position.iloc[:, : maxlen + 2]
        df_position.to_excel('excel/{}.xlsx'.format(name))
        
        #魔改加入每日跌停&停牌持仓
        maxLen = 0
        for chaos in limitdown_stock:
            maxLen = max(maxLen, len(chaos[1]))
        timelist = self.timeline[self.start_index : self.end_index + 1]
        df = pd.DataFrame(index = timelist, columns = range(maxLen))
        for chaos in limitdown_stock:
            chaosLen = len(chaos[1])
            if chaosLen != 0:
                df.loc[chaos[0]][: chaosLen] = chaos[1]
        df.to_excel('excel/{}_limitdown.xlsx'.format(name))

        return df_position, df

        
        
        
    """
    可视化模块
    """
    
    def Plot_analysis(self, matrix_stock, stockindex_daily, bar = True, name = 'Alpha'): #可以选择是否画出柱状图
        
        """这里是将柱状图设为百分数"""
        
        fmt='%.2f%%'
        yticks = mtick.FormatStrFormatter(fmt)
        
        """计算各种参数"""
#       win_stock = self.win_stock[self.start_index : self.end_index + 1]
#       win_daily = [sum(win_stock[i][stockindex_daily[i]]) / (self.pick_end - self.pick_start + 1) for i in range(win_stock.shape[0])]
        equity = matrix_stock[:, 0] #净值列表
        timelist = self.timeline[self.start_index : self.end_index + 1] #绘图时间轴
        returns = np.append(0, equity[1 : ] / equity[ : -1] - 1) #每日收益率
        zz500 = self.zz500_close[self.start_index + 1 : self.end_index + 1] / self.zz500_close[self.start_index : self.end_index] - 1 #中证500每日收益率
        zz500 = np.append(0, zz500.flatten())
        returns_zz500 = returns - zz500 #每日超额收益, 相对中证500
#       winrate = round(len(returns[returns > 0]) / (self.M - 1), 4) #总胜率
        winrate_zz500 = round(len(returns_zz500[returns_zz500 > 0]) / (self.M - 1), 4) #总胜率, 相对中证500
        returnD = round(equity[-1] ** (1 / (self.M - 1)) - 1, 4) #每日平均收益率
        volatilityD = round(np.std(returns), 4) #每日波动率，是每日的标准差
        sharperatio = round((equity[-1] ** (250 / (self.M - 1)) - 1 - 0.03) / (volatilityD * 250 ** (0.5)), 2)
        def MaxDD(equity): #计算最大回撤的函数
            top, down = 0, 1 #定义两个指针,down永远在top前面
            pos = [top,down] #最大回撤位置
            maxDD = (equity[top] / equity[down] - 1)            
            while(down < self.M - 1):
                down += 1
#               maxDD = max(maxDD, equity[top] / equity[down] - 1)
                if maxDD <= 1 - equity[down] / equity[top]:
                    pos = [top,down]
                    maxDD = 1 - equity[down] / equity[top]
                if equity[down] >= equity[top]:
                    top = down
            return pos, round(maxDD, 4)
        pos, maxDD = MaxDD(equity) #最大回撤位置，最大回撤率
        maxDD_start,maxDD_end = timelist[pos[0]], timelist[pos[1]]
            
        """绘图区 高清颜值版"""
        
        plt.style.use('ggplot') #设置ggplot绘图风格
        fig = plt.figure(figsize = (30, 10))
        ax1 = fig.add_subplot(111)
        ax1.plot(timelist, equity, color = 'blue')
        ax1.set_xticks(range(0, len(timelist), int(len(timelist) / 18)))
        ax1.set_ylabel('Equity')
        ax1.set_title('Model : {}, Holding period = 2 From {} to {} CON'.format(name, self.start_day, self.end_day), fontsize = 25)
        plt.text(self.timeline[self.start_index], max(equity) , 'pick from {} to {}, return = {}, returnD = {}, volatilityD = {}, winrate = {}, sharperatio = {}, maxDD = {} form {} to {} '.format(self.pick_start, self.pick_end, round(equity[-1], 2), returnD, volatilityD, winrate_zz500, sharperatio, maxDD, maxDD_start, maxDD_end), bbox = dict(boxstyle = 'round', alpha = 0.7), fontsize = 15)
        if bar:
            ax2 = ax1.twinx()
            ax2.set_xticks(range(0, len(timelist), int(len(timelist) / 18)))
            ax2.set_ylabel('Daily_return')
            ax2.yaxis.set_major_formatter(yticks)
            ax2.bar(timelist, returns_zz500 * 100)
            
        plt.savefig('figure/{}/{}_{}.jpg'.format(name, name, time.strftime("%Y_%m_%d-%H_%M_%S")), dpi = 300, bbox_inches = 'tight') #名字给定，后面加入系统时间戳保证不覆盖掉原图






