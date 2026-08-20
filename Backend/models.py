
"""
TaskMiner - Database Models
============================
SQLite via SQLAlchemy. One file, zero setup -- good for development and
for a college project demo. Swap the DB_URL in db.py for Postgres later
without touching this file (that's the point of using an ORM).
"""

from sqlalchemy import Column, String, DateTime, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class ProjectORM(Base):
    __tablename__ = "projects"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    # all message texts seen for this project, joined by "\n" -- rebuilt into
    # a list in memory on startup so the TF-IDF matcher has its corpus back
    corpus = Column(Text, default="")


class TaskORM(Base):
    __tablename__ = "tasks"

    id = Column(String, primary_key=True)
    raw_text = Column(Text, nullable=False)
    description = Column(Text, nullable=False)
    project_id = Column(String, nullable=False)
    deadline = Column(DateTime, nullable=True)
    priority = Column(String, default="low")
    status = Column(String, default="todo")       # "todo" | "done"
    alarm_active = Column(String, default="no")    # "no" | "yes" -- simple flag, SQLite-friendly
    created_at = Column(DateTime)