from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware # NUEVO: Importar esto
import joblib
import pandas as pd
import mlflow
import os


# ... resto del código ...

mlflow.set_tracking_uri("databricks")

app = FastAPI(title="API Turismo Canarias (Visitantes)")

URL_MODELO="models:/workspace.default.modelofinal/1"

# NUEVO: Configuración de CORS para que tu HTML pueda hablar con la API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite que cualquiera (tu HTML) se conecte
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    pipeline = mlflow.pyfunc.load_model(URL_MODELO)
    print("✅ Pipeline Master cargado.")
except Exception as e: 
    print(e)
    print("❌ Error cargando pkl")

# ... (El resto de tu código sigue igual: class InputDatos, endpoints, etc.) ...
# Solo asegúrate de copiar la parte de arriba
class InputDatos(BaseModel):
    anio: int
    mes: int
    pais: str
    noches: int
    tipo: str


@app.get("/", response_class=HTMLResponse)
def cargarINDEX():
    return FileResponse("index.html") 

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API de Turismo activa"}

@app.post("/predict")
def predict(data: InputDatos):
    try:
        tipo_oficial = "Turista no residente (no tránsito)"
        
        if "excursionista" in data.tipo.lower():
            tipo_oficial = "Excursionista no residente (no tránsito)"
        elif "transito" in data.tipo.lower() or "tránsito" in data.tipo.lower():
            tipo_oficial = "Turista no residente en tránsito"

        df_input = pd.DataFrame([{
            'anio': data.anio,
            'mes': data.mes,
            'pernoctaciones': data.noches,
            'pais_residencia': data.pais,
            'tipo_visitante': tipo_oficial
        }])
        
        prediccion = pipeline.predict(df_input)[0]
        if prediccion < 0: prediccion = 0
        
        return {
            "prediccion_m_euros": round(prediccion, 2),
            "detalle": f"{data.pais} ({tipo_oficial}) - {data.noches} noches"
        }
    except Exception as e:
        return {"error": str(e)}