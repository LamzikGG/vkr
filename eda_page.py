# eda_page.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def eda_page():
    st.title("Исследовательский анализ данных (EDA) исходного датасета")
    
    # Загрузка данных
    uploaded_file = st.file_uploader("Загрузите CSV-файл с данными", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file) 
            
            # Очистка названий колонок (удаляем квадратные скобки)
            df.columns = [col.replace('[', '').replace(']', '') for col in df.columns]
            
            st.success("Данные успешно загружены.")
            
            st.subheader("Первые строки датасета")
            st.write(df.head())
            
            st.subheader("Основные статистики")
            st.write(df.describe())
            
            st.subheader("Гистограммы распределения числовых признаков")
            numerical_cols = ['Air temperature K', 'Process temperature K',
                            'Rotational speed rpm', 'Torque Nm', 'Tool wear min']
            for col in numerical_cols:
                fig, ax = plt.subplots()
                sns.histplot(df[col], kde=True, ax=ax)
                ax.set_title(f'Распределение {col}')
                st.pyplot(fig)

            # Корреляционная матрица
            st.subheader("Корреляционная матрица")
            corr = df[numerical_cols].corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
            st.pyplot(fig)

            # Scatter plot
            st.subheader("Scatter Plot между признаками")
            col1 = st.selectbox("Выберите первый признак", numerical_cols)
            col2 = st.selectbox("Выберите второй признак", numerical_cols, index=1)
            fig, ax = plt.subplots()
            sns.scatterplot(x=df[col1], y=df[col2], hue=df['Machine failure'], alpha=0.6, ax=ax)
            ax.set_title(f'{col1} vs {col2}')
            st.pyplot(fig)

            # Распределение целевой переменной
            st.subheader("Распределение целевой переменной (Machine failure)")
            fig, ax = plt.subplots()
            sns.countplot(x='Machine failure', data=df, palette="Set2")
            ax.set_xticklabels(['Без отказа', 'Отказ'])
            st.pyplot(fig)

            # Анализ типов отказов
            st.subheader("Анализ типов отказов")
            failure_types = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
            failure_counts = df[failure_types].sum()
            fig, ax = plt.subplots()
            sns.barplot(x=failure_counts.index, y=failure_counts.values, palette="viridis")
            ax.set_ylabel("Частота")
            ax.set_title("Частота различных типов отказов")
            st.pyplot(fig)

            # Зависимость отказов от Tool wear
            st.subheader("Зависимость отказов от износа инструмента")
            fig, ax = plt.subplots()
            sns.boxplot(x='Machine failure', y='Tool wear min', data=df, ax=ax)
            ax.set_xticklabels(['Без отказа', 'Отказ'])
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Ошибка: {e}")
    else:
        st.info("Пожалуйста, загрузите CSV-файл для анализа.")

eda_page()