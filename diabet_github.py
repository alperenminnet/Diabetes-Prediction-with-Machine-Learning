import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from helpers.data_prep import *
from helpers.eda import *



pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

# Veriyi Çekelim

df = pd.read_csv("datasets/diabetes.csv")
df.head()

# Bağımlı değişkeni dışarıda bırakalım

cols = [col for col in df.columns if "Outcome" not in col]

# İlk önce veriyi tanıyalım
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]

    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df) # Kategorik,Nümerik ve Kategorik ama Kardinal değişkenleri belirledik.


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum ())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols: # Nümerik değişkenleri tanıdık.
    num_summary(df,col)


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Outcome", col) # Nümerik değişkenlerin sınıflarının bağımlı değişken ile arasındaki ilişkiyi inceledik.

# Veri ön işleme yapalım

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df, col)) # Sadece İnsulin değişkeninde outlier değer var.

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    if check_outlier(df, col):
        replace_with_thresholds(df, col) # Outlier değerleri baskıladık.

# Modelimizin daha iyi olması için yeni Feature türetelim

#Glucose
df["New_glucose_cat"] = df["Glucose"].apply(lambda x: "Normal" if x < 125 else "Not_normal")

#BloodPressure

# Ortalama olarak küçük tansiyonun ise 70-90 arasında olması normal kabul edilir.

df["New_bloodpressure_cat"] = df["BloodPressure"].apply(lambda x: "High" if x > 90 else "Normal")

#Vke

vke = ["zayıf", "normal", "kilolu", "obez","fazla obez"]
df['vki_cat'] = pd.cut(df['BMI'], [0, 18.5, 25, 30,40, df['BMI'].max()],
                       labels=vke)
df.head()

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Artık Encoding işlemlerine geçebiliriz.

# Label Encoding
def label_encoder(dataframe, binary_col):
    le = LabelEncoder()
    dataframe[binary_col] = le.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

# One-Hot Encoding
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
df = one_hot_encoder(df, ohe_cols, drop_first=True) # Dummy Tuzağına düşmemek için drop_first argümanına True giriyoruz.
df.head()

cat_cols, num_cols, cat_but_car = grab_col_names(df)


rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])

# Model

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)



log_model = LogisticRegression().fit(X_train, y_train)


print('intercept:', log_model.intercept_)
print('coefficients:', log_model.coef_)

# Tahmin Başarısını Değerlendirme #

# Train RMSE
y_pred = log_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))

# TRAIN RKARE
log_model.score(X_train, y_train)

# Test RMSE
y_pred = log_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# Test RKARE
log_model.score(X_test, y_test)

# 10 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(log_model, X, y, cv=10, scoring="neg_mean_squared_error")))

# AUC Score için y_prob (1. sınıfa ait olma olasılıkları)
y_prob = log_model.predict_proba(X_test)[:, 1]

# Classification report
print(classification_report(y_test, y_pred))


# ROC Curve
plot_roc_curve(log_model, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()

roc_auc_score(y_test, y_prob) # 0.8123048668503213 --> Tahmin başarımız.