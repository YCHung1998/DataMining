import os
from utils import *
import warnings
warnings.filterwarnings("ignore")

DATA_ROOT = 'dataset'
FIGURE_ROOT = 'figure'
FILENAME = 'character-deaths.csv'

df = pd.read_csv(os.path.join(DATA_ROOT, FILENAME))
data = DataPreProcess(df)
data.get_data(is_train=True)
# Machine Learning Decision Tree Method
ML_DT = DecisionTree(data)
data.y_train_prediction = ML_DT.model.predict(data.X_train)
data.y_test_prediction = ML_DT.model.predict(data.X_test)
eva_train = Evaluate(data.y_train_prediction, data.y_train)
eva_test = Evaluate(data.y_test_prediction, data.y_test)
eva_test.print_confusion_matrix()
eva_test.print_all()