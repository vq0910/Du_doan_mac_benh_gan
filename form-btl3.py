import pandas as pd
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier

from tkinter import *
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score

df = pd.read_csv("D:\mechineleaning\indian_liver_patient_dataset.csv")
X = df.drop(['class'], axis=1) 
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=3)
pca = PCA(7)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.fit_transform(X_test)
id3 = DecisionTreeClassifier(criterion='entropy', max_depth= 13, random_state=0)
id3.fit(X_train_pca,y_train)
y_pred = id3.predict(X_test_pca)

form = tk.Tk()
form.title("Dự đoán người bị mắc bệnh gan:")
form.geometry("450x750")

lb_ten = Label(form, text = "Nhập thông tin bệnh nhân:")
lb_ten.grid(row = 1, column = 1, padx = 10, pady = 10)

lb_age = Label(form, text = "Tuổi của bệnh nhân tính theo năm (age)")
lb_age.grid(row = 2, column = 1, padx = 10, pady = 10)
tb_age = Entry(form)
tb_age.grid(row = 2, column = 2)

lb_gender = Label(form, text = "Giới tính bệnh nhân: Nam hoặc Nữ (gender)")
lb_gender.grid(row = 3, column = 1, padx = 10, pady = 10)
tb_gender = Entry(form)
tb_gender.grid(row = 3, column = 2)

lb_TB = Label(form, text = "Bilirubin toàn phần (TB)")
lb_TB.grid(row = 4, column = 1,padx = 10, pady = 10)
tb_TB = Entry(form)
tb_TB.grid(row = 4, column = 2)

lb_DB = Label(form, text = "Bilirubin trực tiếp (DB)")
lb_DB.grid(row = 5, column = 1, padx = 10, pady = 10)
tb_DB = Entry(form)
tb_DB.grid(row = 5, column = 2)

lb_alkphos = Label(form, text = "Phosphosite kiềm (alkphos)")
lb_alkphos.grid(row = 6, column = 1, padx = 10, pady = 10 )
tb_alkphos = Entry(form)
tb_alkphos.grid(row = 6, column = 2)

lb_sgpt = Label(form, text = "Alanine Aminotransferase (sgpt)")
lb_sgpt.grid(row = 7, column = 1, padx = 10, pady = 10 )
tb_sgpt = Entry(form)
tb_sgpt.grid(row = 7, column = 2)

lb_sgot = Label(form, text = "Aspartate Aminotransferase (sgot)")
lb_sgot.grid(row = 8, column = 1, padx = 10, pady = 10 )
tb_sgot = Entry(form)
tb_sgot.grid(row = 8, column = 2)

lb_TP = Label(form, text = "Tổng số protein (TP) ")
lb_TP.grid(row = 9, column = 1, padx = 10, pady = 10 )
tb_TP = Entry(form)
tb_TP.grid(row = 9, column = 2)

lb_ALB = Label(form, text = "Albumin (ALB)")
lb_ALB.grid(row = 10, column = 1, padx = 10, pady = 10 )
tb_ALB = Entry(form)
tb_ALB.grid(row = 10, column = 2)

lb_A_G = Label(form, text = "Tỷ lệ Albumin và Globulin (A_G):")
lb_A_G.grid(row =11, column = 1, padx = 10, pady = 10 )
tb_A_G = Entry(form)
tb_A_G.grid(row = 11, column = 2)

def kq():
    buying = tb_age.get()
    maint = tb_gender.get()
    doors = tb_TB.get()
    persons = tb_DB.get()
    lug_boot =tb_alkphos.get()
    safety =tb_sgpt.get()
    sgot =tb_sgot.get()
    TP =tb_TP.get()
    ALB =tb_ALB.get()
    A_G =tb_A_G.get()

    X_pca = np.array([buying,maint,doors,persons,lug_boot,safety,sgot,TP,ALB,A_G],dtype = float).reshape(1, -1)
    X_pca = pca.transform(X_pca)
    y = id3.predict(X_pca)
    lbl.configure(text='Kết quả dự đoán là:'+str(y))

lbl = Label(form, text="")
lbl.grid(column=2, row=12)
Button(form, text='Dự đoán kết quả', command= kq).grid(row=12, column=1, pady=10)

precision = str(round(precision_score(y_test, y_pred,average='macro')*100,1))+ "%"
recall = str(round(recall_score(y_test, y_pred,average='macro')*100,1)) + "%"
f1_score = str(round(f1_score(y_test,y_pred,average = 'macro')*100,1)) + "%"
accuracy_score = str(round(accuracy_score(y_test,y_pred)*100,1)) + "%"

columns = ('stt','Kết quả dự đoán','Kết quả thực')
tree = ttk.Treeview(form, columns=columns, show='headings',height='5')
tree.grid(column=0,row=13,columnspan=3)
tree.column('stt',width=50)
tree.column('Kết quả dự đoán',width=100)
tree.column('Kết quả thực',width=100)
tree.heading('stt',text='stt')
tree.heading('Kết quả dự đoán', text='Kết quả dự đoán')
tree.heading('Kết quả thực', text='Kết quả thực')

dt_array = []
count = y_pred.size

y_test = np.array(y_test)
y_pred = np.array(y_pred)
for i in range(count):
    dt_array.append((i, y_pred[i], y_test[i]))

for dt in dt_array:
    tree.insert('', tk.END, values=dt)

Label(form, text='Đánh giá chất lượng mô hình').grid(column=1, row=14, padx=10,pady=2 ,sticky = "W")
Label(form, text='Accuracy_score: ' + str(accuracy_score)).grid(column=1, row=15, padx=10,pady=2 ,sticky = "W")
Label(form, text='Precision: ' + str(precision)).grid(column=1, row=16, padx=10,pady=2 ,sticky = "W")
Label(form, text='Recall: ' + str(recall)).grid(column=1, row=17, padx=10,pady=2,sticky = "W")
Label(form, text='F1_score: ' + str(f1_score)).grid(column=1, row=18, padx=10,pady=2,sticky = "W")

form.mainloop()