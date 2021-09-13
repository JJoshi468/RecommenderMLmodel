import pandas as pd
import joblib
from sklearn import tree
music_data=pd.read_csv('music.csv')
A=music_data.drop(columns=['genre'])
B=music_data['genre']

model=DecisionTreeClassifier()
model.fit(A,B)
predictions=model.predict([ [21, 1] , [22,0]])

tree.export_graphviz(model, out_file='music-recommender.dot',
                     feature_names=['age', 'gender'], 
                     class_names=sorted(B.unique()),
                    label='all',
                    rounded=True,
                    filled=True)


