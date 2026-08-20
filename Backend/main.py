
"""
TaskMiner - FastAPI Backend
=============================
Wires together: nlp_core (extraction + project clustering), the SQLite DB
(models.py / db.py), and a background scheduler that fires the persistent
pre-deadline alarm.

Run with:  uvicorn main:app --reload
Docs at:   http://127.0.0.1:8000/docs   (FastAPI's auto-generated test UI)
"""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy.orm import Session

from db import init_db, get_session, SessionLocal
from models import ProjectORM, TaskORM
from nlp_core import TaskMinerNLP, Project, Task

app = FastAPI(title="TaskMiner API")

# allow a local frontend (React dev server, or plain HTML file opened in browser)
# to call this API during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# serves index.html at "/" and any other files under static/ -- this means
# ONE Render deployment gives you both the API and the UI on the same URL,
# no separate frontend host needed.
@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")

app.mount("/static", StaticFiles(directory="static"), name="static")

engine = TaskMinerNLP()
scheduler = BackgroundScheduler()


# ---------------------------------------------------------------------------
# Request/response schemas
# ---------------------------------------------------------------------------

class NewTaskRequest(BaseModel):
    text: str
    project_hint: Optional[str] = None


class TaskResponse(BaseModel):
    id: str
    description: str
    project_id: str
    project_name: str
    deadline: Optional[datetime]
    priority: str
    status: str
    alarm_active: bool


def to_response(task: Task) -> TaskResponse:
    project = engine.projects[task.project_id]
    return TaskResponse(
        id=task.id, description=task.description, project_id=task.project_id,
        project_name=project.name, deadline=task.deadline, priority=task.priority,
        status=task.status, alarm_active=task.alarm_active,
    )


# ---------------------------------------------------------------------------
# Persistence helpers (keep the in-memory engine and SQLite in sync)
# ---------------------------------------------------------------------------

def persist_project(session: Session, project: Project):
    row = session.get(ProjectORM, project.id)
    corpus_text = "\n".join(project.corpus)
    if row is None:
        row = ProjectORM(id=project.id, name=project.name, corpus=corpus_text)
        session.add(row)
    else:
        row.corpus = corpus_text
    session.commit()


def persist_task(session: Session, task: Task):
    row = session.get(TaskORM, task.id)
    if row is None:
        row = TaskORM(id=task.id)
        session.add(row)
    row.raw_text = task.raw_text
    row.description = task.description
    row.project_id = task.project_id
    row.deadline = task.deadline
    row.priority = task.priority
    row.status = task.status
    row.alarm_active = "yes" if task.alarm_active else "no"
    row.created_at = task.created_at
    session.commit()


def load_state_from_db():
    """Rebuild the in-memory engine (projects + tasks) from SQLite on startup."""
    session = SessionLocal()
    try:
        for row in session.query(ProjectORM).all():
            engine.projects[row.id] = Project(
                id=row.id, name=row.name,
                corpus=row.corpus.split("\n") if row.corpus else [],
            )
        for row in session.query(TaskORM).all():
            task = Task(
                id=row.id, raw_text=row.raw_text, description=row.description,
                project_id=row.project_id, deadline=row.deadline, priority=row.priority,
                status=row.status, alarm_active=(row.alarm_active == "yes"),
                created_at=row.created_at or datetime.now(),
            )
            engine.tasks[task.id] = task
            if task.id not in engine.projects[task.project_id].task_ids:
                engine.projects[task.project_id].task_ids.append(task.id)
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Alarm scheduling -- fires 15 min before deadline, stays "active" until the
# user calls the dismiss endpoint. A frontend is expected to poll
# GET /alarms/active and keep re-playing a sound/notification for anything
# in that list -- that's what makes it "persistent" rather than a one-shot.
# ---------------------------------------------------------------------------

ALARM_LEAD_MINUTES = 15


def trigger_alarm(task_id: str):
    session = SessionLocal()
    try:
        task = engine.tasks.get(task_id)
        if task and task.status == "todo":
            task.alarm_active = True
            persist_task(session, task)
    finally:
        session.close()


def schedule_alarm_for_task(task: Task):
    if not task.deadline:
        return
    trigger_time = task.deadline - timedelta(minutes=ALARM_LEAD_MINUTES)
    if trigger_time <= datetime.now():
        # deadline is already within the alarm window (or past) -- fire immediately
        trigger_alarm(task.id)
        return
    scheduler.add_job(
        trigger_alarm, "date", run_date=trigger_time,
        args=[task.id], id=f"alarm-{task.id}", replace_existing=True,
    )


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
def on_startup():
    init_db()
    load_state_from_db()
    # re-schedule alarms for any tasks that were still pending from a previous run
    for task in engine.tasks.values():
        if task.status == "todo" and not task.alarm_active:
            schedule_alarm_for_task(task)
    scheduler.start()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/tasks", response_model=TaskResponse)
def add_task(payload: NewTaskRequest, session: Session = Depends(get_session)):
    task = engine.process_message(payload.text, project_hint=payload.project_hint)
    project = engine.projects[task.project_id]
    persist_project(session, project)
    persist_task(session, task)
    schedule_alarm_for_task(task)
    return to_response(task)


@app.get("/tasks/todo", response_model=list[TaskResponse])
def get_todo_list():
    return [to_response(t) for t in engine.todo_list()]


@app.get("/tasks/done", response_model=list[TaskResponse])
def get_done_list():
    return [to_response(t) for t in engine.done_list()]


@app.post("/tasks/{task_id}/done", response_model=TaskResponse)
def mark_task_done(task_id: str, session: Session = Depends(get_session)):
    task = engine.tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    task.status = "done"
    task.alarm_active = False
    scheduler.remove_job(f"alarm-{task_id}", jobstore=None) if scheduler.get_job(f"alarm-{task_id}") else None
    persist_task(session, task)
    return to_response(task)


@app.post("/tasks/{task_id}/undo", response_model=TaskResponse)
def undo_task(task_id: str, session: Session = Depends(get_session)):
    """Moves a task from Done back to To-Do. If its deadline has already
    passed, no alarm is re-scheduled (it would fire instantly on a stale
    deadline) -- it just sits in To-Do until manually marked done again."""
    task = engine.tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    task.status = "todo"
    persist_task(session, task)
    if task.deadline and task.deadline > datetime.now():
        schedule_alarm_for_task(task)
    return to_response(task)


@app.post("/tasks/{task_id}/dismiss-alarm", response_model=TaskResponse)
def dismiss_alarm(task_id: str, session: Session = Depends(get_session)):
    """User manually switches the alarm off (does NOT mark the task done)."""
    task = engine.tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    task.alarm_active = False
    persist_task(session, task)
    return to_response(task)


@app.get("/alarms/active", response_model=list[TaskResponse])
def get_active_alarms():
    """Frontend should poll this every few seconds and keep alerting for
    anything returned here -- that's what makes the alarm 'persistent'."""
    return [to_response(t) for t in engine.tasks.values() if t.alarm_active]


@app.get("/projects")
def list_projects():
    return [
        {"id": p.id, "name": p.name, "task_count": len(p.task_ids)}
        for p in engine.projects.values()
    ]


@app.get("/projects/{project_id}/tasks", response_model=list[TaskResponse])
def get_project_tasks(project_id: str):
    if project_id not in engine.projects:
        raise HTTPException(404, "Project not found")
    return [to_response(t) for t in engine.tasks_by_project(project_id)]