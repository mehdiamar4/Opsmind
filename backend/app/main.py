from fastapi import FastAPI

app = FastAPI(title="OpsMind API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/api/version")
def version() -> dict[str, str]:
    return {"version": "0.1.0"}
