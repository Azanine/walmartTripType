from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from main_tools import *
from model_tools import *

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
PATH = "D:/Documents/Machine Learning/Walmart Trip Type/"
if __name__ == '__main__':
    train_df = pd.read_csv(PATH + "eda final/" + "train_new_temp_sincos_catdum_daydum_4favFL.csv")
    test_df = pd.read_csv(PATH + "eda final/" + "test_new_temp_sincos_catdum_daydum_4favFL.csv")

    x_train_sample, x_test, y_train_sample = split_data(train_df.sample(frac=0.1, axis=0), test_df)
    x_train, x_test, y_train = split_data(train_df, test_df)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

    model = train_tune_model(x_train, y_train, x_val, y_val, x_train_sample, y_train_sample)

    pred = model.predict_proba(x_test)

    export_sub(pred, "hyperopt")
