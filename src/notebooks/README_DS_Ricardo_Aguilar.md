# Fase 2 – Evidencia individual  
**Autor:** Ricardo Aguilar  
**Rol:** Data Scientist  
**Branch:** `ricardo-datascientist`  
**Archivo asociado:** `Run_Experiments_DS_Ricardo_Aguilar.py`

---

## Objetivo
Este documento complementa la entrega de la Fase 1 mostrando de forma clara la participación individual dentro del proyecto.  
El objetivo fue ejecutar y registrar experimentos utilizando **MLflow**, validando la reproducibilidad y comparando el desempeño de distintos modelos de predicción.

---

## Actividades realizadas
- Creación del script `run_experiments_ricardo.py` para ejecutar los experimentos sin modificar el código del equipo.  
- Configuración de MLflow para registrar métricas, parámetros y resultados.  
- Ejecución de dos modelos: **Regresión Lineal** (como línea base) y **Random Forest** (para mejora de desempeño).  
- Revisión del flujo de preprocesamiento y separación de datos en entrenamiento y prueba.  
- Documentación del proceso y organización del código bajo buenas prácticas de MLOps.

---

## Flujo general del trabajo
1. **Carga de datos:** con la clase `DataLoader`, leyendo el archivo `energy_efficiency_modified.csv`.  
2. **Preprocesamiento:** con la clase `DataPreprocessor`, aplicando transformaciones básicas antes del modelado.  
3. **Entrenamiento y evaluación:**  
   - `LinearRegression` para obtener una referencia inicial.  
   - `RandomForestRegressor` con 600 árboles, ajustado para mejorar precisión y estabilidad.  
4. **Registro de experimentos:**  
   - Creación del experimento “Energy Efficiency – Ricardo Aguilar” en MLflow.  
   - Registro de parámetros, métricas y observaciones de cada ejecución.  

---


## Resultados observados
| Modelo | R² aproximado | Observaciones |
|:-------|:---------------:|:--------------|
| **Linear Regression** | 0.97 | Modelo base sencillo y transparente; útil como línea de referencia, aunque limitado para capturar relaciones no lineales. |
| **Random Forest** | 0.988 | Presenta un mejor ajuste y generalización; logra capturar interacciones complejas entre variables sin sobreajustar. |
| **Gradient Boosting** | 0.991 | Ofrece el desempeño más alto; optimiza errores residuales de modelos previos, pero requiere mayor costo computacional. |


---

## Posibles mejoras
- Agregar un tercer modelo (por ejemplo, **XGBoost**) para comparar resultados.  
- Implementar validación cruzada y versionado de datos con **DVC**.  
- Utilizar la interfaz de **MLflow UI** para visualizar métricas y comparaciones.  
- Registrar tiempos de ejecución y consumo de recursos para evaluar eficiencia.

---

## Conclusión
Este trabajo demuestra la participación individual en el desarrollo técnico del proyecto, reforzando la trazabilidad y el uso de herramientas profesionales de MLOps.  
La integración de MLflow permitió evidenciar los resultados de cada modelo y dejar el proyecto listo para futuras fases de optimización y automatización.

# Fase 2 – Experimentos del Científico de Datos
**Autor:** Ricardo Miguel Aguilar Rosas  
**Rol:** Data Scientist  
**Equipo:** 8 – MLOps ENB2012  
**Archivo principal:** `F2_DS_Ricardo_Aguilar.ipynb`

## Descripción
Este notebook documenta el flujo de trabajo individual del rol de Data Scientist:
- Carga y validación del dataset ENB2012  
- Preprocesamiento (limpieza, imputación, estandarización)  
- Entrenamiento de modelos (Linear Regression, Random Forest, Gradient Boosting)  
- Registro y evaluación con MLflow  
- Análisis de resultados y conclusiones  

> *Nota:* Este trabajo fue desarrollado en Google Colab y subido de manera independiente para mantener la integridad del repositorio del equipo.
