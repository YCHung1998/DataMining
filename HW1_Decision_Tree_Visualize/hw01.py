
"""#### 2-4 亂數拆成訓練集(75%)與測試集(25%) 
套用sklearn 的 train_test_split，random_state任意取(這裡取6)
"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=6)

"""### 3)使用scikit-learn的DecisionTreeClassifier進行預測
最大深度設定到4層，避免過深作圖時不方便觀測  
"""

from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import cross_val_score
from sklearn import tree
clf = DecisionTreeClassifier(max_depth=4,random_state=0)
clf = clf.fit(X_train, y_train)
clf

prediction = clf.predict(X_test)
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
fig = plt.figure(figsize=(6, 3), dpi=350)
tp =  tree.plot_tree(clf,  filled=True)

"""#### 由於上述的 decision tree 是針對 index 來顯示，不便閱讀
故使用其他方式來將 feature name 放進圖表裡面，能更方便觀察如何做決策的  
其中裡面會計算 gini值、到子葉的樣本數以及分類數量結果

#### 此處因位在本機端電腦無法順利執行程式，改由使用Google 的 Colab執行，將結果貼至下方表格
分類分別為 Book to Death ，也就是死亡，以及 0 : 代表存活
"""

# import graphviz  
# dot_data = tree.export_graphviz(clf, out_file=None, 
#                   feature_names=list(X_train.columns.values),  
#                   class_names=list(y_train.name),
#                   filled = True,
#                   rounded = True,
#                   special_characters = True)

# graph = graphviz.Source(dot_data)
# graph.render('mygraph',view = True)

# graph = graphviz.Source(dot_data)
# graph

"""![mygraph.png](attachment:mygraph.png)

# Training value
觀察在這個深度下的樹訓練資料精確率以及相關的其他數值呈現
"""

from sklearn.metrics import confusion_matrix
print('Confusion Marix : ')
print(confusion_matrix(clf.predict(X_train), y_train))
tn, fp, fn, tp = confusion_matrix(clf.predict(X_train), y_train).ravel()
accuray = (tp+tn)/(tp+tn+fp+fn)
precision = tp/(tp+fp)
recall = tp/(tp+fn)      
sensitivity = tp/(tp+fn) # TPR
specficity = tn/(tn+fp) # FPR
f1_score = 1/(1/sensitivity+1/specficity)
print(f'''Accuary     = {accuray}, 
Precision   = {precision}, 
Recall      = {recall},
Sensitivity = {sensitivity}, 
Specficity  = {specficity}, 
F1_Score    = {f1_score}.''')

# 在位來可能會使用到，先練習如何使用
from sklearn.metrics import log_loss
log_loss(prediction, y_test)

"""# Testing Value
觀測在前面訓練測資完成後的 Decision Tree，面對未知的問題是否能有效的分類  
一樣是數值結果呈現
"""

from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix(prediction, y_test).ravel()
print('Confusion Marix : ')
print(confusion_matrix(prediction, y_test))
accuray = (tp+tn)/(tp+tn+fp+fn)
precision = tp/(tp+fp)
recall = tp/(tp+fn)      
sensitivity = tp/(tp+fn) # TPR
specficity = tn/(tn+fp) # FPR
f1_score = 1/(1/sensitivity+1/specficity)

print(f'''Accuary     = {accuray}, 
Precision   = {precision}, 
Recall      = {recall},
Sensitivity = {sensitivity}, 
Specficity  = {specficity}, 
F1_Score    = {f1_score}.''')

"""### 對Testing set 利用熱擴散圖畫出混淆矩陣數量分布
上方橫列代表的是 Ground Truth，側邊縱列代表的是 Prediction  
0 : 表示存活， 1 : 表示死亡  
(最原始圖效果不佳，加上對應的表格數量值，顏色軸，說明所佔的數量分別為何)
"""

import seaborn as sns
# confusion_matrix(prediction, y_test),
# 上橫 : GT , 側縱 : Pred
print('上方橫列代表的是 Ground Truth，側邊縱列代表的是 Prediction.')
print('0 : 表示存活， 1 : 表示死亡')
fig =sns.heatmap(confusion_matrix(prediction, y_test),
                  cmap = 'Spectral', # Spectral, YlOrRd
                  annot = True,
                  annot_kws={"size":30},
                  fmt = 'd',
                  linewidths = 1.5,
                  linecolor = 'Black')