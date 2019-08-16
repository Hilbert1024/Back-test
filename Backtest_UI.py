# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 18:01:35 2019

@author: Hilbert
"""
import numpy as np
import tkinter
from tkinter import filedialog
import Backtest

root = tkinter.Tk()
# root.withdraw()
root.title("回测系统 V1.0")
root.geometry("400x400+200+50")
title = tkinter.Label(root, text="回测系统 V1.0", fg="black", font=("微软雅黑", 20))
title.pack()

grade = tkinter.Label(root, text="选择因子", fg="black", font=("微软雅黑", 15))
grade.place(x = 10, y = 50)

def SelectGrade():
    global gradePath
    gradePath = filedialog.askopenfilenames(title = '选择因子得分', filetypes = [('numpy data',('.npy'))])[0]
    return

gradeSelect = tkinter.Button(root, text = "选择", command = SelectGrade, width = 8)
gradeSelect.place(x = 120, y = 50)
grade_start_input = tkinter.Variable()
grade_start_input.set('2010-01-04')
startTimeInput = tkinter.Entry(root, textvariable = grade_start_input, width = 20)
startTimeInput.place(x = 200, y = 55)

startTime = tkinter.Label(root, text="开始时间", fg="black", font=("微软雅黑", 15))
startTime.place(x = 10, y = 100)
start_time_input = tkinter.Variable()
startTimeInput = tkinter.Entry(root, textvariable = start_time_input, width = 20)
startTimeInput.place(x = 120, y = 105)

endTime = tkinter.Label(root, text="结束时间", fg="black", font=("微软雅黑", 15))
endTime.place(x = 10, y = 150)
end_time_input = tkinter.Variable()
endTimeInput = tkinter.Entry(root, textvariable = end_time_input, width = 20)
endTimeInput.place(x = 120, y = 155)

startPick = tkinter.Label(root, text="起始选取", fg="black", font=("微软雅黑", 15))
startPick.place(x = 10, y = 200)
start_pick_input = tkinter.Variable()
start_pick_input.set(1)
startPickInput = tkinter.Entry(root, textvariable = start_pick_input, width = 20)
startPickInput.place(x = 120, y = 205)

endPick = tkinter.Label(root, text="结束选取", fg="black", font=("微软雅黑", 15))
endPick.place(x = 10, y = 250)
end_pick_input = tkinter.Variable()
end_pick_input.set(50)
endPickInput = tkinter.Entry(root, textvariable = end_pick_input, width = 20)
endPickInput.place(x = 120, y = 255)

modelName = tkinter.Label(root, text="模型名称", fg="black", font=("微软雅黑", 15))
modelName.place(x = 10, y = 300)
model_name_input = tkinter.Variable()
model_name_input.set("Default Model")
modelNameInput = tkinter.Entry(root, textvariable = model_name_input, width = 20)
modelNameInput.place(x = 120, y = 305)

def Backtest_UI():
    grade = np.load(gradePath)
    start_day = start_time_input.get()
    end_day = end_time_input.get()
    pick_start = int(start_pick_input.get())
    pick_end = int(end_pick_input.get())
    names = model_name_input.get()
    #
    print(start_day,end_day,pick_start,pick_end,names)
    BT = Backtest.Back_test(start_day, end_day, pick_start, pick_end)
        
    BT.Data_load()
        
    stockindex_daily = BT.Select_stock_array(grade, grade_start_input.get(), ST = False)

    matrix_stock, limitdown_stock = BT.Back_test_mode(stockindex_daily)
        
    # position = BT.Save_data(matrix_stock, limitdown_stock, name = names)
    #
    BT.Plot_analysis(matrix_stock, stockindex_daily, bar = True, name = names)

    print("Back test finish!")

startTest = tkinter.Button(root, text = "开始回测", command = Backtest_UI, width = 20, height = 0)
startTest.place(x = 120, y = 355)

root.mainloop()
