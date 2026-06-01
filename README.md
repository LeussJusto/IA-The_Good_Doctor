# The Good Doctor

## Sistema Inteligente de Diagnóstico Clínico mediante Deep Learning

Proyecto de Machine Learning y Deep Learning orientado a la detección temprana de Sepsis y clasificación de condiciones clínicas utilizando pipelines multicapa, ingeniería de características y arquitecturas neuronales progresivas.

---

# Descripción General

The Good Doctor es un proyecto enfocado en el análisis clínico predictivo utilizando datos biomédicos estructurados.

El objetivo principal es construir modelos capaces de:

* Detectar Sepsis tempranamente
* Clasificar condiciones médicas críticas
* Reducir falsos negativos clínicos
* Mejorar recall en enfermedades de alto riesgo
* Simular razonamiento médico mediante pipelines Two-Stage

---
# Dataset y Modelos: 

Dataset: https://www.kaggle.com/datasets/uom190346a/synthetic-clinical-tabular-dataset

Modelos: https://huggingface.co/leuss1224/TheGoodDoctor_Models

# Arquitectura General del Proyecto

```text
The_Good_Doctor/
│
├── DeepLearning_v1/
│   ├── v1/
│   ├── v2/
│   ├── v3/
│   ├── v4/
│   ├── v5/
│   └── v6/
│
├── DeepLearning_v2/   (En desarrollo)
│
├── datasets/
├── modelos/
├── logs/

```

---

# Entorno de Desarrollo

| Componente | Detalle              |
| ---------- | -------------------- |
| Plataforma | Google Colab         |
| Framework  | TensorFlow / Keras   |
| Lenguaje   | Python               |
| Dataset    | clinical_dataset.csv |

---

# Dataset Clínico

El dataset contiene variables biomédicas y clínicas utilizadas para clasificación multiclase de condiciones médicas.

## Variables Principales

* age
* bmi
* heart_rate
* systolic_bp
* diastolic_bp
* glucose
* creatinine
* smoker
* diabetes
* diagnosis

---

# DeepLearning v1

Primera generación del pipeline clínico.

Esta fase documenta la evolución progresiva del sistema desde modelos básicos de Machine Learning hasta arquitecturas neuronales multicapa especializadas en detección de Sepsis.

---

# Evolución de Versiones — DeepLearning v1

| Versión | Arquitectura          | Objetivo Principal             |
| ------- | --------------------- | ------------------------------ |
| v1      | Baseline ML           | Establecer benchmark inicial   |
| v2      | Random Forest + SMOTE | Mejorar Recall de Sepsis       |
| v3      | MLP Básico            | Introducción a Deep Learning   |
| v4      | MLP Multiclase        | Clasificación clínica completa |
| v5      | Pipeline Two-Stage    | Separación Sepsis vs No-Sepsis |
| v6      | Two-Stage Optimizado  | Corrección Feature Engineering |

---

# v1 — Baseline Machine Learning

## Objetivo

Construir una línea base utilizando modelos clásicos de Machine Learning para medir el rendimiento mínimo aceptable.

## Modelos Explorados

* DummyClassifier
* Logistic Regression
* Random Forest

## Hallazgos

* Accuracy alta pero Recall de Sepsis extremadamente bajo
* El desbalance de clases afecta severamente el diagnóstico clínico
* Accuracy no es suficiente en contexto médico

---

# v2 — Random Forest + SMOTE

## Objetivo

Combatir el desbalance de clases utilizando oversampling sintético.

## Técnica Implementada

SMOTE (Synthetic Minority Oversampling Technique)

## Beneficios

* Incremento significativo del Recall de Sepsis
* Mejor representación de clases minoritarias

## Limitaciones

* Reducción de Accuracy global
* Riesgo de sobreajuste en datos sintéticos
* F1 Macro todavía insuficiente para producción clínica

---

# v3 — MLP Básico

## Objetivo

Migrar desde Machine Learning clásico hacia Deep Learning.

## Arquitectura

Multilayer Perceptron (MLP):

* Dense Layers
* ReLU
* Dropout
* Softmax

## Mejoras

* Mayor capacidad de representación
* Mejor aprendizaje de patrones no lineales
* Soporte nativo para class weights

---

# v4 — MLP Multiclase

## Objetivo

Clasificación simultánea de múltiples condiciones médicas:

* Normal
* Heart Failure
* Pneumonia
* Sepsis

## Problema Detectado

El modelo tenía dificultades para detectar Sepsis debido al desbalance extremo.

## Hallazgos

* Recall de Sepsis insuficiente
* Muchos falsos negativos clínicamente peligrosos
* Necesidad de rediseñar la arquitectura del pipeline

---

# v5 — Pipeline Two-Stage

## Innovación Principal

Separación del problema en dos etapas:

### Stage 1

Clasificador binario:

* Sepsis
* No-Sepsis

### Stage 2

Clasificador multiclase:

* Normal
* Heart Failure
* Pneumonia

## Ventajas

* Especialización total en detección de Sepsis
* Mayor Recall clínico
* Mejor manejo del desbalance extremo

## Problema Detectado

Una feature derivada presentaba una desviación estándar incorrecta (~3.235), afectando severamente el rendimiento del Stage 1.

---

# v6 — Two-Stage Optimizado

## Mejora Crítica

Corrección completa del Feature Engineering.

## Features Derivadas

| Feature                | Significado Clínico            |
| ---------------------- | ------------------------------ |
| pulse_pressure         | Diferencia de presión arterial |
| map_approx             | Presión Arterial Media         |
| log_creatinine_glucose | Índice renal/metabólico        |
| organ_risk_score_v6    | Riesgo de fallo orgánico       |

## Mejoras Implementadas

* Corrección matemática de features derivadas
* RobustScaler para variables clínicas
* Mejor estabilidad del Stage 1
* Reducción significativa de falsos positivos

## Resultados

| Métrica Clínica  | Objetivo |
| ---------------- | -------- |
| Recall Sepsis    | ≥ 0.55   |
| F1 Sepsis        | ≥ 0.25   |
| Precision Sepsis | ≥ 0.12   |

---

# Arquitectura Neuronal

## MLP Utilizado

| Capa    | Configuración |
| ------- | ------------- |
| Input   | n_features    |
| Dense 1 | 128 + ReLU    |
| Dense 2 | 64 + ReLU     |
| Dense 3 | 32 + ReLU     |
| Output  | Softmax       |

## Regularización

* Batch Normalization
* Dropout
* EarlyStopping
* Class Weights

---

# Técnicas Implementadas

## Balanceo de Clases

* SMOTE
* Class Weights
* Two-Stage Pipeline

## Preprocesamiento

* LabelEncoder
* StandardScaler
* RobustScaler

## Feature Engineering

* Variables clínicas derivadas
* Indicadores Sepsis-3
* Riesgo orgánico compuesto

---

# Métricas Clínicas

| Métrica           | Importancia              |
| ----------------- | ------------------------ |
| Recall Sepsis     | Métrica más crítica      |
| F1 Sepsis         | Balance precisión-recall |
| F1 Macro          | Rendimiento general      |
| Balanced Accuracy | Robustez ante desbalance |
| FN Sepsis         | Riesgo clínico real      |

En contexto médico, los False Negatives son el error más crítico, ya que representan pacientes con Sepsis no detectados.

---

# DeepLearning v2 (En Desarrollo)

Segunda generación del sistema clínico.

Esta fase busca reconstruir el pipeline utilizando arquitecturas más avanzadas orientadas a producción médica.

---

# Objetivos de DeepLearning v2

* Mejorar Recall de Sepsis
* Reducir falsos negativos críticos
* Implementar modelos híbridos
* Incorporar interpretabilidad clínica
* Optimizar inferencia para despliegue

---

# Tecnologías Planeadas

| Área              | Tecnología            |
| ----------------- | --------------------- |
| Deep Learning     | TensorFlow / PyTorch  |
| Arquitecturas     | TabNet / Transformers |
| Interpretabilidad | SHAP / LIME           |
| Deployment        | FastAPI               |
| Serving           | Docker                |
| Inferencia        | ONNX                  |

---

# Roadmap

## DeepLearning v1

* [x] Baseline ML
* [x] Random Forest + SMOTE
* [x] Migración a MLP
* [x] Clasificación multiclase
* [x] Pipeline Two-Stage
* [x] Optimización Feature Engineering

## DeepLearning v2

* [ ] Arquitecturas avanzadas
* [ ] Interpretabilidad clínica
* [ ] Fine-tuning médico
* [ ] Optimización productiva
* [ ] API clínica
* [ ] Dashboard médico

---

# Futuro del Proyecto

El objetivo final es construir un sistema capaz de:

* Asistir diagnóstico clínico temprano
* Detectar pacientes de alto riesgo
* Reducir mortalidad por Sepsis
* Apoyar decisiones médicas
* Integrarse en sistemas hospitalarios

---

# Licencia

Proyecto desarrollado con fines educativos, investigación y experimentación en inteligencia artificial aplicada a salud.
