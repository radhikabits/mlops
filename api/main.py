"""FastAPI Module"""
from fastapi import FastAPI
from router import agent
 
 
# Create FastAPI app
app = FastAPI(
    title="Prediction API",
    description="API for model prediction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)
 
app.include_router(agent.router)
 
 
# Health check endpoint
@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}