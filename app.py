import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Cargar modelo y scaler
modelo = joblib.load('model/modelo_logistico.pkl')
scaler = joblib.load('model/scaler.pkl')  # si lo usaste

st.title("🧠 Dashboard de Segmentación de Clientes - LTV")

st.markdown("Completa los datos del cliente para predecir si tiene un LTV **alto** o **bajo**.")

# Inputs del usuario (ajusta con tus variables reales)
age = st.number_input("Edad", min_value=18, max_value=100, value=30)
active_days = st.number_input("Días activo", min_value=0, value=50)
cashback = st.number_input("Cashback recibido", min_value=0.0, value=100.0)
satisfaction = st.slider("Satisfacción del cliente (1-5)", 1, 5, 3)
resolution_time = st.number_input("Tiempo de resolución de incidencias (días)", min_value=0, value=5)
last_txn = st.number_input("Días desde última transacción", min_value=0, value=15)
loyalty_points = st.number_input("Puntos de lealtad ganados", min_value=0, value=100)
referrals = st.number_input("Número de referidos", min_value=0, value=2)
support_tickets = st.number_input("Tickets de soporte levantados", min_value=0, value=1)
max_txn = st.number_input("Valor máximo de transacción", min_value=0.0, value=500.0)
min_txn = st.number_input("Valor mínimo de transacción", min_value=0.0, value=10.0)
location_woe = st.slider("WOE de la ubicación", min_value=-0.5, max_value=0.5, value=0.0)
income = st.selectbox("Nivel de ingreso", ['Bajo', 'Medio', 'Alto'])
usage = st.selectbox("Frecuencia de uso de la app", ['Bajo', 'Medio', 'Alto'])
payment = st.selectbox("Método de pago preferido", ['Débito', 'Crédito', 'Transferencia', 'Efectivo'])

# WoE valores predefinidos (los que usaste en el entrenamiento)
woe_dict = {
    'Income_Level': {'Bajo': -0.1, 'Medio': 0.0, 'Alto': 0.2},
    'App_Usage_Frequency': {'Bajo': -0.3, 'Medio': 0.0, 'Alto': 0.3},
    'Preferred_Payment_Method': {'Débito': 0.1, 'Crédito': -0.1, 'Transferencia': 0.2, 'Efectivo': -0.2}
}


# Todas las columnas que el modelo espera (orden exacto del entrenamiento)
columnas_modelo = [
    'Age', 'Active_Days', 'Cashback_Received', 'Customer_Satisfaction_Score',
    'Issue_Resolution_Time', 'Last_Transaction_Days_Ago', 'Loyalty_Points_Earned',
    'Referral_Count', 'Support_Tickets_Raised',
    'Max_Transaction_Value', 'Min_Transaction_Value', 'Location_WOE',
    'Income_Level_WOE', 'App_Usage_Frequency_WOE', 'Preferred_Payment_Method_WOE'
]

# Crear diccionario con todos los valores en 0 por defecto
data_dict = {col: 0 for col in columnas_modelo}


# Rellenar con los valores del usuario
data_dict['Age'] = age
data_dict['Active_Days'] = active_days
data_dict['Cashback_Received'] = cashback
data_dict['Customer_Satisfaction_Score'] = satisfaction
data_dict['Issue_Resolution_Time'] = resolution_time
data_dict['Last_Transaction_Days_Ago'] = last_txn
data_dict['Loyalty_Points_Earned'] = loyalty_points
data_dict['Referral_Count'] = referrals
data_dict['Support_Tickets_Raised'] = support_tickets
data_dict['Max_Transaction_Value'] = max_txn
data_dict['Min_Transaction_Value'] = min_txn
data_dict['Location_WOE'] = location_woe
data_dict['Income_Level_WOE'] = woe_dict['Income_Level'][income]
data_dict['App_Usage_Frequency_WOE'] = woe_dict['App_Usage_Frequency'][usage]
data_dict['Preferred_Payment_Method_WOE'] = woe_dict['Preferred_Payment_Method'][payment]


# Convertir a DataFrame
data = pd.DataFrame([data_dict])
data = data.loc[:, modelo.feature_names_in_]

threshold = st.slider("🔧 Umbral para clasificar LTV Alto", 0.0, 1.0, 0.5)

# Botón para hacer la predicción
if st.button("🔍 Predecir LTV"):
    # Escalar si corresponde
    X = scaler.transform(data)

    # Predicción
    prob = modelo.predict_proba(X)[0][1]
    pred = 1 if prob >= threshold else 0

    # Mostrar resultados
    st.markdown("---")
    st.subheader("🔎 Resultado de la Predicción")
    st.write(f"**LTV Predicho:** {'Alto (1)' if pred == 1 else 'Bajo (0)'}")
    st.write(f"**Probabilidad de ser LTV Alto:** {prob:.2%}")

    # Interpretación del riesgo
    if prob >= 0.75:
        st.success("💰 Este cliente tiene un LTV muy alto. ¡Altamente valioso!")
    elif prob >= 0.5:
        st.info("📈 Este cliente tiene un LTV alto. Vale la pena enfocarse en él.")
    elif prob >= 0.3:
        st.warning("⚠️ Este cliente tiene un LTV medio. Se recomienda monitorear.")
    else:
        st.error("🔻 Este cliente tiene un LTV bajo. Riesgo de bajo retorno.")

    # Barra de progreso visual
    st.progress(min(int(prob * 100), 100))

    # Visual adicional (barra de riesgo)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 1))
    ax.barh(['Probabilidad LTV Alto'], [prob], color='orange')
    ax.set_xlim(0, 1)
    ax.set_xlabel("Nivel de riesgo")
    st.pyplot(fig)