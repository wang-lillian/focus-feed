from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from services.search import search
from services.process_articles import index_documents
import uvicorn

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def render_home(request: Request):
    return templates.TemplateResponse(request=request, name="home.html")


@app.post("/read", response_class=HTMLResponse)
def render_read(request: Request, user_interest: str = Form(...)):
    index_documents(user_interest)
    results = search(user_interest)
    return templates.TemplateResponse(
        request=request,
        name="read.html",
        context={"articles": results, "user_interest": user_interest},
    )


if __name__ == "__main__":
    uvicorn.run(app)
