from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="API Turismo Canarias (Visitantes)")

try:
    pipeline = joblib.load("pipeline_turismo.pkl")
    print("✅ Pipeline Master cargado.")
except:
    print("❌ Error cargando pkl")

class InputDatos(BaseModel):
    anio: int
    mes: int
    pais: str
    noches: int
    tipo: str  # Nuevo campo (Ej: "Turista" o "Excursionista")

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API de Turismo activa y modelo cargado correctamente"}

@app.post("/predict")
def predict(data: InputDatos):
    try:
        # Mapeamos lo que escriba el usuario a los nombres oficiales del dataset
        # Esto hace la API más fácil de usar
        tipo_oficial = "Turista no residente (no tránsito)" # Valor por defecto
        
        if "excursionista" in data.tipo.lower():
            tipo_oficial = "Excursionista no residente (no tránsito)"
        elif "transito" in data.tipo.lower() or "tránsito" in data.tipo.lower():
            tipo_oficial = "Turista no residente en tránsito"

        # Creamos el DF con las 5 variables
        df_input = pd.DataFrame([{
            'anio': data.anio,
            'mes': data.mes,
            'pernoctaciones': data.noches,
            'pais_residencia': data.pais,
            'tipo_visitante': tipo_oficial # Usamos el nombre técnico
        }])
        
        prediccion = pipeline.predict(df_input)[0]
        if prediccion < 0: prediccion = 0
        
        return {
            "prediccion_m_euros": round(prediccion, 2),
            "detalle": f"{data.pais} ({tipo_oficial}) - {data.noches} noches"
        }
    except Exception as e:
        return {"error": str(e)}