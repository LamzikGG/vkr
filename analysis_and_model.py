# analysis_and_model.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

def analysis_and_model_page():
    st.title("Анализ данных и предсказание отказов оборудования")

    uploaded_file = st.file_uploader("Загрузите CSV-файл с данными", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Удаление ненужных столбцов
            df = df.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], errors='ignore')

            # Преобразование категориальной переменной Type
            le = LabelEncoder()
            df['Type'] = le.fit_transform(df['Type'])

            # Проверка пропущенных значений
            if df.isnull().sum().any():
                st.warning("Обнаружены пропущенные значения. Они будут удалены.")
                df = df.dropna()
            
            # Очистка названий колонок (удаляем квадратные скобки)
            df.columns = [col.replace('[', '').replace(']', '') for col in df.columns]
            
            # Выбор признаков и целевой переменной
            X = df.drop(columns=['Machine failure'])
            y = df['Machine failure']

            # Масштабирование числовых признаков
            scaler = StandardScaler()
            numerical_cols = ['Air temperature K', 'Process temperature K',
                            'Rotational speed rpm', 'Torque Nm', 'Tool wear min']
            X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

            # Разделение на обучающую и тестовую выборки
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            st.success("Данные успешно загружены и обработаны.")

            # Выбор модели
            model_choice = st.selectbox("Выберите модель машинного обучения",
                                    ["Логистическая регрессия", "Случайный лес", "XGBoost"])

            if st.button("🚀 Обучить модель"):
                if model_choice == "Логистическая регрессия":
                    model = LogisticRegression(max_iter=1000)
                elif model_choice == "Случайный лес":
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                else:
                    from xgboost import XGBClassifier
                    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]

                acc = accuracy_score(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=False)
                auc = roc_auc_score(y_test, y_proba)

                st.subheader("Результаты модели")
                st.write(f"**Accuracy**: {acc:.2f}")
                st.write(f"**ROC AUC**: {auc:.2f}")

                st.subheader("Classification Report")
                st.text(report)

                st.subheader("Confusion Matrix")
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                st.pyplot(fig)

                # ROC Curve
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                fig_roc, ax_roc = plt.subplots()
                ax_roc.plot(fpr, tpr, label=f'{model_choice} (AUC = {auc:.2f})')
                ax_roc.plot([0, 1], [0, 1], 'k--')
                ax_roc.set_xlabel('False Positive Rate')
                ax_roc.set_ylabel('True Positive Rate')
                ax_roc.set_title('ROC Curve')
                ax_roc.legend()
                st.pyplot(fig_roc)

                # Сохранение модели в session_state
                st.session_state.model = model
                st.session_state.scaler = scaler
                st.session_state.le = le

            # Предсказание по новым данным
            st.subheader("🔮 Предсказание по новым данным")
            with st.form("prediction_form"):
                product_type = st.selectbox("Тип продукта (L, M, H)", ['L', 'M', 'H'])
                air_temp = st.number_input("Температура окружающей среды [K]", value=120.0)
                process_temp = st.number_input("Температура процесса [K]", value=409.0)
                rotational_speed = st.number_input("Скорость вращения [rpm]", value=1000)
                torque = st.number_input("Крутящий момент [Nm]", value=69.0)
                tool_wear = st.number_input("Износ инструмента [мин]", value=500)

                submitted = st.form_submit_button("Предсказать")

                if submitted and 'model' in st.session_state:
                    model = st.session_state.model
                    scaler = st.session_state.scaler
                    le = st.session_state.le

                    input_data = pd.DataFrame({
                        'Type': [le.transform([product_type])[0]],
                        'Air temperature K': [air_temp],
                        'Process temperature K': [process_temp],
                        'Rotational speed rpm': [rotational_speed],
                        'Torque Nm': [torque],
                        'Tool wear min': [tool_wear]
                    })

                    input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])
                    prediction = model.predict(input_data)[0]
                    probability = model.predict_proba(input_data)[0][1]

                    st.success(f"Оборудование {'откажет' if prediction == 1 else 'не откажет'}")
                    st.info(f"Вероятность отказа: {probability:.2%}")

        except Exception as e:
            st.error(f"Ошибка: {e}")

analysis_and_model_page()