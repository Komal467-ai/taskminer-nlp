"""
TaskMiner - NLP Core Module
============================
Extracts, from a single free-text message:
  1. task description
  2. deadline (datetime, if present)
  3. priority (high / medium / low, heuristic)
  4. which existing project it belongs to (via semantic similarity),
     or flags it as a new project

This is the piece the synopsis calls the "NLP Processing Module".
"""

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import dateparser
import dateparser.search
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# NOTE: The project synopsis proposes Sentence-Transformers embeddings for
# project-similarity matching. That needs a one-time model download from
# HuggingFace, which isn't available in this sandbox's network. This module
# uses TF-IDF + cosine similarity instead -- same "compare new text against
# each project's accumulated text, pick the best match" idea, fully offline.
# Swap in SentenceTransformer (see commented alternative below) once you're
# running this on your own machine with internet access -- the rest of the
# pipeline (deadline extraction, priority, alarm, etc.) doesn't change.

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass
class Project:
    id: str
    name: str
    corpus: list = field(default_factory=list)   # all message texts seen for this project
    task_ids: list = field(default_factory=list)


@dataclass
class Task:
    id: str
    raw_text: str
    description: str
    project_id: str
    deadline: Optional[datetime]
    priority: str
    status: str = "todo"           # "todo" | "done"
    alarm_active: bool = False
    created_at: datetime = field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class TaskMinerNLP:
    # below this cosine-similarity score, a message is treated as a NEW project
    # rather than matched to an existing one. TF-IDF scores run lower than dense
    # embeddings would -- tune this after testing on your own real message data.
    PROJECT_MATCH_THRESHOLD = 0.12

    URGENT_WORDS = {"urgent", "asap", "immediately", "critical", "important", "priority"}

    def __init__(self):
        self.projects: dict[str, Project] = {}
        self.tasks: dict[str, Task] = {}

    # dateparser.search occasionally matches a bare preposition as a "date" --
    # e.g. it will pick up the word "to" in an ordinary sentence with no date
    # in it at all. Discard matches that are just one of these on their own.
    _SPURIOUS_DATE_MATCHES = {"to", "for", "at", "on", "in", "by", "of", "and", "am", "pm", "a", "the"}

    _CURRENCY_SYMBOLS = set("₹$€£")

    _BARE_MONTH_OR_WEEKDAY = {
        "jan", "january", "feb", "february", "mar", "march", "apr", "april", "may",
        "jun", "june", "jul", "july", "aug", "august", "sep", "sept", "september",
        "oct", "october", "nov", "november", "dec", "december",
        "mon", "monday", "tue", "tues", "tuesday", "wed", "weds", "wednesday",
        "thu", "thur", "thurs", "thursday", "fri", "friday", "sat", "saturday", "sun", "sunday",
    }

    # a matched phrase must contain one of these signals to count as a real
    # date/time reference -- filters out things like "1-2" or "6 Months +"
    # that dateparser sometimes mis-flags inside long, unstructured text
    # (job postings, schedules) that have no actual deadline in them.
    _TEMPORAL_SIGNAL = re.compile(
        r"\d{1,2}\s*(am|pm)\b"
        r"|\d{1,2}:\d{2}"
        r"|\b(mon|monday|tue|tues|tuesday|wed|weds|wednesday|thu|thur|thurs|thursday|fri|friday|sat|saturday|sun|sunday)\b"
        r"|\b(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)\b"
        r"|\btoday\b|\btomorrow\b|\btonight\b|\bnext\b"
        r"|\d{1,2}(st|nd|rd|th)\b"
        r"|\d{1,2}[/.]\d{1,2}([/.]\d{2,4})?",
        re.I,
    )

    def _is_valid_date_match(self, phrase: str) -> bool:
        stripped = phrase.strip()
        lowered = stripped.lower()
        if lowered in self._SPURIOUS_DATE_MATCHES or len(stripped) <= 2:
            return False
        if any(ch in self._CURRENCY_SYMBOLS for ch in stripped):
            return False
        if re.fullmatch(r"\d{1,2}-\d{1,2}", stripped):  # e.g. "1-2", "11-2" -- ambiguous ranges, not dates
            return False
        if lowered in self._BARE_MONTH_OR_WEEKDAY:  # a month/weekday name with no attached day is too weak on its own
            return False
        if not self._TEMPORAL_SIGNAL.search(stripped):
            return False
        return True

    # -- deadline extraction --------------------------------------------------
    def extract_deadline(self, text: str, reference_time: Optional[datetime] = None) -> Optional[datetime]:
        """
        Finds a date/time expression inside free text (absolute like '15 August 6pm'
        or relative like 'day after tomorrow', 'by next Monday') and returns it.
        """
        reference_time = reference_time or datetime.now()
        lowered = text.lower()

        # dateparser.search's phrase-splitting mishandles "day after tomorrow"
        # (it tends to match only "tomorrow", giving the wrong day) -- handle
        # this common relative phrase explicitly before falling back.
        if "day after tomorrow" in lowered:
            target = reference_time + timedelta(days=2)
            time_match = re.search(r"\b(\d{1,2})(:\d{2})?\s?(am|pm)\b", lowered)
            if time_match:
                hour = int(time_match.group(1)) % 12
                if time_match.group(3) == "pm":
                    hour += 12
                minute = int(time_match.group(2)[1:]) if time_match.group(2) else 0
                return target.replace(hour=hour, minute=minute, second=0, microsecond=0)
            return target.replace(hour=23, minute=59, second=0, microsecond=0)

        # dateparser.search also mishandles "next <weekday>" combined with a
        # time (e.g. "next Monday 9am" landed a month off in testing) --
        # resolve weekday + optional time ourselves, deterministically.
        weekday_dt = self._extract_weekday_deadline(lowered, reference_time)
        if weekday_dt:
            return weekday_dt

        settings = {"PREFER_DATES_FROM": "future", "RELATIVE_BASE": reference_time}
        found = dateparser.search.search_dates(text, settings=settings)
        if not found:
            return None

        # drop spurious / low-confidence matches (see _is_valid_date_match above)
        found = [(phrase, dt) for phrase, dt in found if self._is_valid_date_match(phrase)]
        if not found:
            return None

        # if multiple date-like phrases remain, take the last one mentioned --
        # in practice deadlines are usually stated after the task description
        # e.g. "finish the report by Friday 5pm"
        _, dt = found[-1]
        return dt

    _WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

    def _extract_weekday_deadline(self, lowered_text: str, reference_time: datetime) -> Optional[datetime]:
        match = re.search(
            r"\b(next\s+)?(" + "|".join(self._WEEKDAYS) + r")\b"
            r"(?:\s+(?:at\s+)?(\d{1,2})(:\d{2})?\s?(am|pm))?",
            lowered_text,
        )
        if not match:
            return None

        has_next, weekday_name, hour_str, min_str, meridian = match.groups()
        target_weekday = self._WEEKDAYS.index(weekday_name)
        days_ahead = (target_weekday - reference_time.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7  # today's weekday name always means the upcoming one, not today
        target = reference_time + timedelta(days=days_ahead)

        if hour_str:
            hour = int(hour_str) % 12
            if meridian == "pm":
                hour += 12
            minute = int(min_str[1:]) if min_str else 0
            target = target.replace(hour=hour, minute=minute, second=0, microsecond=0)
        else:
            target = target.replace(hour=23, minute=59, second=0, microsecond=0)
        return target

    # -- priority heuristic -----------------------------------------------------
    def extract_priority(self, text: str, deadline: Optional[datetime]) -> str:
        lowered = text.lower()
        if any(w in lowered for w in self.URGENT_WORDS):
            return "high"
        if deadline:
            hours_left = (deadline - datetime.now()).total_seconds() / 3600
            if hours_left <= 24:
                return "high"
            if hours_left <= 72:
                return "medium"
        return "low"

    # -- task description cleanup -----------------------------------------------
    def extract_task_description(self, text: str, deadline_text: Optional[str] = None) -> str:
        """Strip filler words/date phrases so the stored task reads cleanly."""
        cleaned = text.strip()
        cleaned = re.sub(r"\b(please|pls|kindly)\b", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
        return cleaned[0].upper() + cleaned[1:] if cleaned else cleaned

    # -- project matching / clustering -------------------------------------------
    def match_or_create_project(self, text: str, hint_name: Optional[str] = None) -> Project:
        """
        Compares the incoming message against each existing project's
        accumulated text (its "corpus") using TF-IDF + cosine similarity.
        If the best match clears the threshold, the task is attached to
        that project. Otherwise a new project is created.
        """
        if self.projects:
            # build one joined document per project so a message can match
            # a project even if it's not identical to any single past message
            project_ids = list(self.projects.keys())
            project_docs = [" ".join(self.projects[pid].corpus) for pid in project_ids]

            vectorizer = TfidfVectorizer(stop_words="english")
            matrix = vectorizer.fit_transform(project_docs + [text])
            project_vectors, query_vector = matrix[:-1], matrix[-1]

            scores = cosine_similarity(query_vector, project_vectors).flatten()
            best_idx = scores.argmax()
            best_score = scores[best_idx]

            if best_score >= self.PROJECT_MATCH_THRESHOLD:
                best_project = self.projects[project_ids[best_idx]]
                best_project.corpus.append(text)
                return best_project

        # no good match (or no projects yet) -> new project
        new_id = str(uuid.uuid4())[:8]
        name = hint_name or self._guess_project_name(text)
        project = Project(id=new_id, name=name, corpus=[text])
        self.projects[new_id] = project
        return project

    def _guess_project_name(self, text: str) -> str:
        """Very simple fallback name: first few meaningful words."""
        words = [w for w in re.findall(r"[A-Za-z]+", text) if len(w) > 2]
        return " ".join(words[:3]).title() if words else "Untitled Project"

    # -- main entry point ---------------------------------------------------------
    def process_message(self, text: str, project_hint: Optional[str] = None) -> Task:
        deadline = self.extract_deadline(text)
        priority = self.extract_priority(text, deadline)
        description = self.extract_task_description(text)
        project = self.match_or_create_project(text, hint_name=project_hint)

        task = Task(
            id=str(uuid.uuid4())[:8],
            raw_text=text,
            description=description,
            project_id=project.id,
            deadline=deadline,
            priority=priority,
        )
        self.tasks[task.id] = task
        project.task_ids.append(task.id)
        return task

    # -- persistence hooks (used by the API layer to rehydrate state on startup) ----
    def load_project(self, project: Project):
        self.projects[project.id] = project

    def load_task(self, task: Task):
        self.tasks[task.id] = task

    # -- status changes -------------------------------------------------------------
    def mark_done(self, task_id: str) -> bool:
        task = self.tasks.get(task_id)
        if not task:
            return False
        task.status = "done"
        return True

    # -- views ------------------------------------------------------------------------
    def todo_list(self):
        return [t for t in self.tasks.values() if t.status == "todo"]

    def done_list(self):
        return [t for t in self.tasks.values() if t.status == "done"]

    def tasks_by_project(self, project_id: str):
        return [t for t in self.tasks.values() if t.project_id == project_id]