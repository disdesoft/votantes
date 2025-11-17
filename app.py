# ===========================================================
#  API DE PREDICCIÓN DE INTENCIÓN DE VOTO - KNN
#  Servicio REST con FastAPI
# ===========================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===========================================================
# INICIALIZAR FASTAPI
# ===========================================================
app = FastAPI(
    title="API de Predicción de Intención de Voto",
    description="Modelo KNN para predecir intención de voto basado en características demográficas",
    version="1.0.0"
)

# Configurar CORS para permitir peticiones desde el frontend
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica los dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================================================
# MODELOS DE DATOS (PYDANTIC)
# ===========================================================
class VotanteInput(BaseModel):
    """Modelo de entrada para predicción"""
    age: Optional[int] = None
    gender: Optional[int] = None
    education: Optional[int] = None
    employment_status: Optional[int] = None
    employment_sector: Optional[int] = None
    income_bracket: Optional[int] = None
    marital_status: Optional[int] = None
    household_size: Optional[int] = None
    has_children: Optional[int] = None
    urbanicity: Optional[int] = None
    region: Optional[int] = None
    voted_last: Optional[int] = None
    party_id_strength: Optional[int] = None
    union_member: Optional[int] = None
    public_sector: Optional[int] = None
    home_owner: Optional[int] = None
    small_biz_owner: Optional[int] = None
    owns_car: Optional[int] = None
    wa_groups: Optional[int] = None
    refused_count: Optional[int] = None
    attention_check: Optional[int] = None
    will_turnout: Optional[float] = None
    undecided: Optional[int] = None
    preference_strength: Optional[int] = None
    survey_confidence: Optional[int] = None
    tv_news_hours: Optional[int] = None
    social_media_hours: Optional[int] = None
    trust_media: Optional[int] = None
    civic_participation: Optional[int] = None
    job_tenure_years: Optional[int] = None
    primary_choice: Optional[str] = None
    secondary_choice: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "age": 35,
                "gender": 1,
                "education": 3,
                "income_bracket": 2,
                "urbanicity": 1,
                "party_id_strength": 4,
                "voted_last": 1
            }
        }

class PrediccionOutput(BaseModel):
    """Modelo de salida para predicción"""
    prediccion: str
    probabilidades: Dict[str, float]
    confianza: float
    mejor_k: int

class ModeloInfo(BaseModel):
    """Información del modelo"""
    algoritmo: str
    parametros: Dict
    accuracy: float
    clases: List[str]
    features_numericas: int
    features_categoricas: int
    total_entrenamiento: int

# ===========================================================
# VARIABLES GLOBALES PARA EL MODELO
# ===========================================================
modelo = None
columnas_numericas = None
columnas_categoricas = None
clases = None
mejor_k = None
accuracy_test = None
X_cols = None

# ===========================================================
# FUNCIÓN DE ENTRENAMIENTO
# ===========================================================
def entrenar_modelo():
    """Entrena el modelo KNN con el dataset"""
    global modelo, columnas_numericas, columnas_categoricas, clases, mejor_k, accuracy_test, X_cols
    
    logger.info("Iniciando entrenamiento del modelo...")
    
    try:
        # Cargar dataset
        df = pd.read_csv("voter_intentions_3000.csv")
        logger.info(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
        
        objetivo = "intended_vote"
        
        # Limpiar datos
        df = df.dropna(subset=[objetivo])
        
        X = df.drop(columns=[objetivo])
        y = df[objetivo]
        
        # Guardar nombres de columnas
        X_cols = X.columns.tolist()
        
        # Detectar columnas numéricas y categóricas
        columnas_numericas = X.select_dtypes(include=["int64", "float64"]).columns
        columnas_categoricas = X.select_dtypes(include=["object"]).columns
        
        logger.info(f"Columnas numéricas: {len(columnas_numericas)}")
        logger.info(f"Columnas categóricas: {len(columnas_categoricas)}")
        
        # Preprocesamiento
        preprocesamiento = ColumnTransformer([
            ("numericas", Pipeline([
                ("imputar", SimpleImputer(strategy="mean")),
                ("escalar", StandardScaler())
            ]), columnas_numericas),
            
            ("categoricas", Pipeline([
                ("imputar", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), columnas_categoricas)
        ])
        
        # Pipeline con KNN
        pipeline_modelo = Pipeline([
            ("prep", preprocesamiento),
            ("knn", KNeighborsClassifier())
        ])
        
        # Parámetros para GridSearch
        parametros = {
            "knn__n_neighbors": [3, 5, 7, 9],
            "knn__weights": ["uniform", "distance"],
            "knn__p": [1, 2]
        }
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )
        
        logger.info("Ejecutando GridSearchCV...")
        busqueda = GridSearchCV(pipeline_modelo, parametros, cv=5, n_jobs=-1, verbose=0)
        busqueda.fit(X_train, y_train)
        
        modelo = busqueda.best_estimator_
        mejor_k = busqueda.best_params_['knn__n_neighbors']
        clases = modelo.classes_.tolist()
        
        # Calcular accuracy
        accuracy_test = modelo.score(X_test, y_test)
        
        logger.info(f"Modelo entrenado exitosamente")
        logger.info(f"Mejor K: {mejor_k}")
        logger.info(f"Accuracy: {accuracy_test:.4f}")
        logger.info(f"Clases: {clases}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error durante el entrenamiento: {str(e)}")
        return False

# ===========================================================
# ENDPOINTS DE LA API
# ===========================================================

@app.on_event("startup")
async def startup_event():
    """Entrenar modelo al iniciar la API"""
    logger.info("Iniciando API...")
    success = entrenar_modelo()
    if success:
        logger.info("✓ Modelo listo para recibir peticiones")
    else:
        logger.error("✗ Error al entrenar el modelo")

@app.get("/api")
def root():
    """Endpoint raíz"""
    return {
        "mensaje": "API de Predicción de Intención de Voto",
        "version": "1.0.0",
        "estado": "activo" if modelo is not None else "entrenando",
        "endpoints": {
            "health": "/health",
            "info": "/info",
            "predecir": "/predecir",
            "docs": "/docs"
        }
    }

@app.get("/api/health")
def health_check():
    """Verificar estado del servicio"""
    return {
        "status": "healthy" if modelo is not None else "training",
        "modelo_cargado": modelo is not None
    }

@app.get("/api/info", response_model=ModeloInfo)
def obtener_info():
    """Obtener información del modelo"""
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    return ModeloInfo(
        algoritmo="K-Nearest Neighbors (KNN)",
        parametros={
            "n_neighbors": mejor_k,
            "metric": "euclidean",
            "weights": "uniform"
        },
        accuracy=float(accuracy_test),
        clases=clases,
        features_numericas=len(columnas_numericas),
        features_categoricas=len(columnas_categoricas),
        total_entrenamiento=2400  # 80% de 3000
    )

@app.post("/api/predecir", response_model=PrediccionOutput)
def predecir(votante: VotanteInput):
    """Predecir intención de voto"""
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    try:
        # Convertir a DataFrame
        datos = votante.dict()
        df_votante = pd.DataFrame([datos])
        
        # Hacer predicción
        prediccion = modelo.predict(df_votante)[0]
        probabilidades = modelo.predict_proba(df_votante)[0]
        
        # Crear diccionario de probabilidades
        probs_dict = {clase: float(prob) for clase, prob in zip(clases, probabilidades)}
        confianza = float(max(probabilidades))
        
        return PrediccionOutput(
            prediccion=prediccion,
            probabilidades=probs_dict,
            confianza=confianza,
            mejor_k=mejor_k
        )
        
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error al procesar datos: {str(e)}")

@app.post("/api/predecir_batch")
def predecir_batch(votantes: List[VotanteInput]):
    """Predecir múltiples votantes a la vez"""
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    try:
        resultados = []
        
        for votante in votantes:
            datos = votante.dict()
            df_votante = pd.DataFrame([datos])
            
            prediccion = modelo.predict(df_votante)[0]
            probabilidades = modelo.predict_proba(df_votante)[0]
            
            probs_dict = {clase: float(prob) for clase, prob in zip(clases, probabilidades)}
            
            resultados.append({
                "prediccion": prediccion,
                "probabilidades": probs_dict,
                "confianza": float(max(probabilidades))
            })
        
        return {
            "total": len(resultados),
            "predicciones": resultados
        }
        
    except Exception as e:
        logger.error(f"Error en predicción batch: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error al procesar datos: {str(e)}")

# ===========================================================
# EJECUTAR LA API
# ===========================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
