import numpy as np
import os
import Backtest

filename = 'random'
grade = np.load('test_data/{}.npy'.format(filename))
start_day = '2010-01-05'
end_day = '2018-12-27'
grade_start = '2010-01-04'
pick_start = 1
pick_end = 50
names = filename

if not os.path.exists('figure/{}/'.format(names)):
    os.mkdir('figure/{}/'.format(names))
#
BT = Backtest.Back_test(start_day, end_day, pick_start, pick_end)

BT.Data_load()

stockindex_daily = BT.Select_stock_array(grade, grade_start, ST = False)

matrix_stock, limitdown_stock = BT.Back_test_mode(stockindex_daily)

# position, _ = BT.Save_data(matrix_stock, limitdown_stock, name = names)

BT.Plot_analysis(matrix_stock, stockindex_daily, bar = True, name = names)

