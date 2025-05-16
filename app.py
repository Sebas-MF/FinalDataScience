import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

@st.cache_resource
def load_model():
    import pickle
    with open('model_rf.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_importances():
    # Asume que el modelo RF regressor tiene feature_importances_
    import pickle
    model = load_model()
    import pandas as pd
    # Nombres de columnas en el mismo orden que X_reg_train
    feature_names = model.feature_names_in_
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    return importances
importances = load_importances()

model = load_model()

st.title("Predicción de LTV de billetera digital")

st.header("Ingrese datos del cliente para la predicción")
age = st.number_input('Edad del cliente', min_value=18, max_value=100, value=30)
income_level = st.selectbox('Nivel de ingreso', ['Bajo', 'Medio', 'Alto'])
usage_freq = st.selectbox('Frecuencia de uso de la app', ['Diario', 'Semanal', 'Mensual'])
preferred_method = st.selectbox('Método de pago preferido', ['Tarjeta', 'Billetera', 'Transferencia'])

show_advanced = st.checkbox('Ingresar detalles avanzados (opcional)')
if show_advanced:
    total_tx = st.number_input('¿Cuántas compras has hecho en total?', min_value=0, value=0)
    avg_tx_value = st.number_input('¿Cuánto gastas en promedio por compra? (Q)', min_value=0.0, value=0.0)
    max_tx_value = st.number_input('¿Cuál fue tu compra más cara? (Q)', min_value=0.0, value=0.0)
    min_tx_value = st.number_input('¿Cuál fue tu compra más barata? (Q)', min_value=0.0, value=0.0)
    active_days = st.number_input('¿Cuántos días llevas usando la app?', min_value=0, value=0)
    days_since_last = st.number_input('¿Hace cuántos días fue tu última compra?', min_value=0, value=0)
    issue_time = st.number_input('Tiempo de resolución de incidencias (horas)', min_value=0.0, value=0.0)

if st.button("Predecir LTV"):
    input_data = {
        'Age': age,
        'Income_Level': income_level,
        'App_Usage_Frequency': usage_freq,
        'Preferred_Payment_Method': preferred_method
    }
    if show_advanced:
        input_data.update({
            'Total_Transactions': total_tx,
            'Avg_Transaction_Value': avg_tx_value,
            'Active_Days': active_days,
            'Last_Transaction_Days_Ago': days_since_last,
            'Issue_Resolution_Time': issue_time
        })
    input_df = pd.DataFrame([input_data])
    # Cargar medianas históricas para imputation de variables faltantes
    df_hist = pd.read_csv('data/digital_wallet_ltv_dataset.csv')
    medians = df_hist.select_dtypes(include='number').median()
    # Asegurar columnas y rellenar con medianas históricas
    input_df = input_df.reindex(columns=model.feature_names_in_)
    input_df = input_df.fillna(medians[input_df.columns])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicción de LTV: {prediction:.2f}")

    # Evaluar nivel de riesgo según cuantiles de LTV
    q1 = df_hist['LTV'].quantile(0.33)
    q2 = df_hist['LTV'].quantile(0.66)
    if prediction < q1:
        st.error("Nivel de riesgo: ALTO 🔴")
    elif prediction < q2:
        st.warning("Nivel de riesgo: MEDIO 🟠")
    else:
        st.success("Nivel de riesgo: BAJO 🟢")

    st.subheader("Importancia de características")
    fig, ax = plt.subplots()
    sns.barplot(x=importances.values, y=importances.index, ax=ax)
    st.pyplot(fig)

    st.subheader("Distribución histórica de LTV")
    fig2, ax2 = plt.subplots()
    sns.histplot(df_hist['LTV'], bins=30, kde=True, ax=ax2)
    st.pyplot(fig2)
