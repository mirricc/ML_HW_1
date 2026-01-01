import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df_train = pd.read_csv('models/df_train_processed.csv')

num_col = df_train.select_dtypes(include=['number']).columns.tolist()

if not num_col:
    st.warning("В датасете нет числовых колонок.")
else:
    st.subheader("Гистограммы числовых признаков")
    cols_to_plot = num_col[:9]
    n = len(cols_to_plot)
    rows = (n + 2) // 3 
    fig_hist, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows))
    
    if rows == 1:
        axes = [axes] if n <= 3 else axes  
    axes = axes.flatten()

    for i, col in enumerate(cols_to_plot):
        sns.histplot(df_train[col].dropna(), bins=30, kde=True, ax=axes[i])
        axes[i].set_title(f'Распределение: {col}', fontsize=12)
        axes[i].set_xlabel(col, fontsize=10)
        axes[i].set_ylabel('Частота', fontsize=10)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig_hist)

    if len(num_col) > 9:
        st.info(f"Показаны первые 9 из {len(num_col)} числовых признаков.")

    st.subheader("Тепловая карта корреляции")
    corr_matrix = df_train[num_col].corr()
    fig_size = max(8, len(num_col) * 0.6)
    fig_corr, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    ax.set_title("Тепловая карта корреляции", fontsize=14, pad=20)
    plt.tight_layout()
    st.pyplot(fig_corr)

model_path = 'models/linear_model.pkl'

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    if not hasattr(model, 'coef_') or not hasattr(model, 'intercept_'):
        st.error("Загруженный объект не является линейной моделью (нет атрибутов coef_ или intercept_).")
    else:
        if 'selling_price' in num_col:
            num_col.remove('selling_price')
        coeffs = model.coef_
        intercept = model.intercept_

        if len(coeffs) != len(num_col):
            st.error(
                f"Несоответствие: модель имеет {len(coeffs)} коэффициентов, "
                f"но признаков — {len(num_col)}. Убедитесь, что модель обучена на тех же данных."
            )
        else:
            feature_importance = pd.DataFrame({
                'feature': num_col,
                'coefficient': coeffs
            })
            feature_importance['abs_coefficient'] = feature_importance['coefficient'].abs()
            feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False)

            st.subheader("Веса (коэффициенты) линейной модели")
            fig_model, ax_model = plt.subplots(figsize=(10, max(6, len(num_col) * 0.3)))
            sns.barplot(data=feature_importance, x='coefficient', y='feature', ax=ax_model, palette='coolwarm')
            ax_model.set_title('Важность признаков (коэффициенты линейной модели)', fontsize=14)
            ax_model.set_xlabel('Коэффициент')
            plt.tight_layout()
            st.pyplot(fig_model)
            st.dataframe(feature_importance[['feature', 'coefficient']].style.format({'coefficient': '{:.4f}'}))

except Exception as e:
    st.error(f"Ошибка при загрузке или обработке модели: {e}")




st.header("Предсказать цену автомобиля")
FEATURES = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque', 'seats', 'max_torque_rpm']
TARGET = 'selling_price'

input_data = {}
cols = st.columns(2)

for i, feature in enumerate(FEATURES):
    col = cols[i % 2]
    default_val = float(df_train[feature].median()) if not df_train[feature].isna().all() else 0.0
    if feature in ['seats']:
        default_val = int(default_val)
        input_data[feature] = col.number_input(f"{feature}", value=default_val, min_value=1, step=1)
    else:
        input_data[feature] = col.number_input(f"{feature}", value=default_val, format="%.2f")
if st.button("Рассчитать цену"):
    try:
        input_data['torque']=np.log1p(input_data['torque'])
        input_data['km_driven']=np.log1p(input_data['km_driven'])
        input_data['year']=input_data['year']**2
        X_input = np.array([[input_data[f] for f in FEATURES]])
        predicted = np.expm1(model.predict(X_input)[0])
        st.success(f"_Предсказанная цена: **{predicted:,.0f}**_")
        #st.metric(label="Прогнозируемая цена", value=f"{predicted:,.0f}")
    except Exception as e:
        st.error(f"Ошибка при предсказании: {e}")
