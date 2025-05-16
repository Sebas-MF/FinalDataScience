# Proyecto FinTech: Predicción de Lifetime Value (LTV)

**Autores:**  
- Luis Pedro González  
- Sebastián Mollinedo Figueroa  

**Fecha:** Mayo 2025  

---

## 1. Descripción

Este proyecto implementa un flujo completo de ciencia de datos para **predecir y clasificar el Lifetime Value (LTV)** de clientes de una billetera digital FinTech. Incluye:

- **Exploración de datos (EDA)**  
- **Modelado de regresión** para pronosticar LTV como valor continuo  
- **Modelado de clasificación** para segmentar clientes en “bajo”, “medio” y “alto” valor  
- **Selección de características** y diagnóstico de fugas de información  
- **Visualización** de la separación de clases (PCA)  
- **Despliegue** mediante una aplicación web interactiva con Streamlit

El objetivo final es proporcionar a marketing, finanzas y riesgos una herramienta basada en datos para optimizar campañas, retención de clientes y gestión de líneas de crédito.

---

## 2. Estructura del repositorio

FinalDataScience/
├── data/
│   └── digital_wallet_ltv_dataset.csv    # Dataset original
├── notebooks/
│   └── LTV.ipynb                         # Notebook con todo el análisis y modelado
├── app.py                                # Streamlit web app para interactuar con el modelo
├── model_rf.pkl                          # Modelo Random Forest Regressor serializado
├── requirements.txt                      # Dependencias del proyecto
└── README.md                             # Este archivo de documentación

---

## 3. Requisitos e instalación

1. Clona este repositorio:
   ```bash
   git clone https://tu-repositorio.git
   cd FinalDataScience
   
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

pandas
numpy
scikit-learn
matplotlib
seaborn
streamlit
reportlab


4. Uso del notebook
	1.	Abre notebooks/LTV.ipynb en JupyterLab o Jupyter Notebook.
	2.	Ejecuta las celdas en orden para:
	•	Cargar y explorar los datos.
	•	Limpiar y preparar el dataset.
	•	Entrenar modelos de regresión y clasificación.
	•	Evaluar y diagnosticar resultados.
	•	Ajustar hiperparámetros con validación cruzada.
	•	Visualizar separación de clases con PCA.
	•	Guardar el modelo final (model_rf.pkl).
	3.	Revisa los gráficos y métricas en cada sección para entender el desempeño de los modelos.

⸻

5. Despliegue de la web app

La aplicación permite que cualquier usuario ingrese información de un cliente y obtenga:
	•	Predicción de LTV (valor continuo).
	•	Nivel de riesgo (alto, medio, bajo) según cuantiles de LTV.
	•	Gráfico de importancias de variables.
	•	Distribución histórica de LTV.

Para levantar la aplicación:

streamlit run app.py

Abre tu navegador en http://localhost:8501.


6. Detalle de cada sección
	1.	EDA
	•	Garantiza comprensión del dataset.
	•	Identifica outliers, asimetrías y relaciones entre variables.
	2.	Modelado de regresión
	•	Regresión lineal como baseline (R² ≈ 0.85).
	•	Random Forest Regressor para captar no-linealidades (R² ≈ 0.9995).
	3.	Clasificación
	•	Logistic Regression balanceada (F1-macro ≈ 0.85).
	•	Random Forest Classifier (F1-macro ≈ 0.92).
	4.	Selección de características
	•	Corr(variables, LTV) y mutual information.
	•	Coeficientes lineales e importancias de RF.
	5.	Visualización PCA
	•	Muestra agrupamiento claro de clientes por categoría.
	6.	Despliegue
	•	App Streamlit con inputs amigables, imputación y visualizaciones.

⸻

7. Contribuciones
	•	Luis Pedro González: limpieza de datos, modelado de regresión, visualizaciones.
	•	Sebastián Mollinedo Figueroa: clasificación, selección de características, despliegue de la web app.

⸻

8. Próximos pasos
	1.	Añadir nuevas variables predictivas (churn rate, actividad en la app).
	2.	Implementar sistema de retraining automático.
	3.	Desplegar con Docker y añadir dashboard de métricas.