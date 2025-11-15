import uvicorn

def start_server(host="127.0.0.1", port=8000):
    """Start the FastAPI server."""
    uvicorn.run("energy_efficiency.api:app", host=host, port=port, reload=True)

if __name__ == "__main__":
    start_server()
