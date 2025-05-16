import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

ruta_modelo = "notebooks/outputs/modelos_guardados/logreg_ltv_opt.pkl"

try:
    modelo = joblib.load(ruta_modelo)
except FileNotFoundError:
    st.error(f"No se encontró el archivo del modelo en: {ruta_modelo}")
    st.stop()

# Título del dashboard
st.title("Predicción del Nivel de LTV de Clientes Fintech")

st.markdown("""
Este modelo predice si un cliente tiene un nivel de LTV **bajo**, **medio** o **alto** 
en función de sus características transaccionales clave.
""")

# Inputs del usuario
total_spent = st.number_input("Total Spent", min_value=0.0)
avg_trans = st.number_input("Avg Transaction Value", min_value=0.0)
total_trans = st.number_input("Total Transactions", min_value=0)
max_trans = st.number_input("Max Transaction Value", min_value=0.0)
min_trans = st.number_input("Min Transaction Value", min_value=0.0)

# Botón para predecir
if st.button("Predecir LTV"):
    # Crear un DataFrame con los valores ingresados
    X_nuevo = pd.DataFrame([[total_spent, avg_trans, total_trans, max_trans, min_trans]],
                           columns=['Total_Spent', 'Avg_Transaction_Value', 'Total_Transactions',
                                    'Max_Transaction_Value', 'Min_Transaction_Value'])

    # Realizar la predicción
    pred = modelo.predict(X_nuevo)

    # Mostrar resultado
    niveles = ['bajo', 'medio', 'alto']
    st.success(f"🔍 Nivel de LTV Predicho: **{niveles[int(pred[0])]}**")

    # Comparar valores ingresados vs. promedio de referencia
    st.markdown("### 📊 Comparación con el perfil promedio")
    promedios = {
        'Total_Spent': 900.0,
        'Avg_Transaction_Value': 80.0,
        'Total_Transactions': 10,
        'Max_Transaction_Value': 200.0,
        'Min_Transaction_Value': 5.0
    }

    df_comp = pd.DataFrame({
        'Variable': list(promedios.keys()),
        'Usuario': [total_spent, avg_trans, total_trans, max_trans, min_trans],
        'Promedio': list(promedios.values())
    })

    df_comp.set_index('Variable').plot(kind='bar', figsize=(8, 5))
    plt.title('Usuario vs Promedio (valores transaccionales)')
    plt.ylabel('Valor')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

    # Guardar la predicción en un DataFrame para descargar
    historial_df = X_nuevo.copy()
    historial_df["Prediccion_LTV"] = niveles[int(pred[0])]

    # Descargar la predicción como CSV
    st.download_button(
        label="📥 Descargar esta predicción como CSV",
        data=historial_df.to_csv(index=False),
        file_name="mi_prediccion.csv",
        mime="text/csv"
    )

    st.markdown("### 🔍 Variables más influyentes del modelo")
    try:
        coef = modelo.coef_[0]
        coef_df = pd.DataFrame({
            'Variable': X_nuevo.columns,
            'Importancia': coef
        }).sort_values(by='Importancia', key=abs, ascending=False)
        st.dataframe(coef_df.head(5))
    except:
        st.warning("Este modelo no permite mostrar importancia de variables directamente.")

    # Comparación con el historial si existe
    if os.path.exists("outputs/historial_predicciones.csv"):
        st.markdown("### 📊 Comparación con otras predicciones")
        historial = pd.read_csv("outputs/historial_predicciones.csv")
        st.write("Distribución de Total Spent por nivel de LTV:")

        import seaborn as sns
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        sns.boxplot(data=historial, x="Prediccion_LTV", y="Total_Spent", ax=ax)
        st.pyplot(fig)

st.markdown("### 🗑 Limpiar Historial de Predicciones")
if st.button("Borrar historial"):
    try:
        os.remove("outputs/historial_predicciones.csv")
        st.success("Historial eliminado correctamente.")
    except FileNotFoundError:
        st.info("No hay historial para eliminar.")