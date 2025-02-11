import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def load_and_clean_data(filepath):
    # Чтение файла и базовая очистка
    df = pd.read_csv(filepath, sep=',')
    df = pd.DataFrame(df)
    df = df.drop(columns=['SellerName', 'StreetName', 'Zipcode', 'DealType', 'VIN', 'Stock#', 'Engine', 'ExteriorColor'])
    
    # Замена "Not Priced" на NaN и преобразование Price в числовой тип
    df['Price'] = df['Price'].replace('Not Priced', np.nan)
    df = df.dropna()
    df['Price'] = df['Price'].str.replace(r'[\$,]', '', regex=True).astype(float)
    
    # Создание нового признака: возраст автомобиля
    df['CarAge'] = 2022 - df['Year']
    scaler = MinMaxScaler()
    df[['CarAge']] = scaler.fit_transform(df[['CarAge']])
    df = df.drop(columns=['Year'])
    
    # Логарифмическое преобразование
    df['Mileage'] = np.log1p(df['Mileage'])
    df['Price'] = np.log1p(df['Price'])
    
    # Очистка от выбросов
    Q1 = df['Price'].quantile(0.001)
    Q3 = df['Price'].quantile(0.9987)
    df = df[(df['Price'] >= Q1) & (df['Price'] <= Q3)]
    
    # Преобразование категориальных признаков
    df = pd.get_dummies(df, columns=['Make', 'Drivetrain'], drop_first=True)
    le = LabelEncoder()
    # Пример target encoding для Model
    df['Model_target'] = df.groupby('Model')['Price'].transform('mean')
    df = df.drop(columns=['Model'])
    df['Used/New'] = le.fit_transform(df['Used/New'])
    df['SellerType'] = le.fit_transform(df['SellerType'])
    df['State'] = le.fit_transform(df['State'])
    df['InteriorColor'] = le.fit_transform(df['InteriorColor'])
    df['FuelType'] = le.fit_transform(df['FuelType'])
    df['Transmission'] = le.fit_transform(df['Transmission'])
    
    return df

def train_model(df):
    # Разделение выборки
    X = df.drop(columns=['Price'])
    y = df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Обучение модели RandomForest
    model = RandomForestRegressor(random_state=42, n_estimators=155, max_depth=20)
    model.fit(X_train, y_train)
    
    # Предсказание и инвертирование логарифмического преобразования
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_test_orig = np.expm1(y_test)
    
    # Вычисление метрик
    mae = mean_absolute_error(y_test_orig, y_pred)
    mse = mean_squared_error(y_test_orig, y_pred)
    r2 = r2_score(y_test_orig, y_pred)
    
    print("MAE:", mae)
    print("MSE:", mse)
    print("R²:", r2)
    
    return model, X_test, y_test_orig, y_pred

def plot_results(y_test, y_pred):
    # График предсказанных цен против реальных
    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.xlabel("Реальная цена")
    plt.ylabel("Предсказанная цена")
    plt.title("Реальная цена vs. Предсказанная цена")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.show()

if __name__ == "__main__":
    df = load_and_clean_data('data\cars_raw.csv')
    model, X_test, y_test_orig, y_pred = train_model(df)
    plot_results(y_test_orig, y_pred)