from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
from apscheduler.schedulers.background import BackgroundScheduler
from threading import Lock
import uvicorn
from services.search import search
from services.process_articles import index_documents


interest_lock = Lock()
latest_user_interest = None
scheduler = BackgroundScheduler()

def scheduled_indexing():
    with interest_lock:
        if latest_user_interest:
            print(f"[Scheduler] Indexing more documents for: {latest_user_interest}")
            index_documents(latest_user_interest)


@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.add_job(scheduled_indexing, "interval", hours=1)
    scheduler.start()
    print("Scheduler started at startup")
    yield
    scheduler.shutdown()
    print("Scheduler shut down")


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def render_home(request: Request):
    return templates.TemplateResponse(request=request, name="home.html")


@app.post("/read", response_class=HTMLResponse)
def render_read(request: Request, user_interest: str = Form(...), action: str = Form(...)):
    global latest_user_interest
    if action == "submit":
        with interest_lock:
            latest_user_interest = user_interest.strip().lower()
        index_documents(user_interest)
    elif action == "refresh":
        print("refreshed!")

    results = search(user_interest)
    return templates.TemplateResponse(
        request=request,
        name="read.html",
        context={"articles": results, "user_interest": user_interest},
    )



if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)
