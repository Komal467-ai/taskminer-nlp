## 🧩 TaskMiner — NLP-Based Task, Project & Deadline Mining System

An NLP-powered task manager that reads plain, everyday text — not structured forms — and automatically extracts the task, identifies which project it belongs to, pulls out the deadline, and enforces it with a persistent alarm that only stops when you dismiss it.

🔗 Live Demo: _deploy in progress — see "Deployment" section below_
🔗 GitHub: https://github.com/Komal467-ai/taskminer-nlp

### Overview

Daily task data usually arrives messy — a quick message, a forwarded schedule, a line buried in a job posting. Most task managers require you to manually create a task, tag a project, and set a reminder. TaskMiner instead:

- Reads free-form text and extracts the **task**, its **deadline**, and its **priority**
- Automatically figures out **which project** the task belongs to — even if related messages are typed on different days, worded differently
- Fires a **persistent pre-deadline alarm** (15 minutes before) that keeps alerting until manually dismissed, instead of a passive notification that's easy to ignore
- Tracks everything in a **To-Do / Done** board, plus a **Projects view** showing per-project progress

### Results

| Check | Outcome |
|---|---|
| Deadline extraction | Correctly resolves absolute ("15 August 6pm"), relative ("day after tomorrow"), and weekday-based ("next Monday 9am") expressions |
| False-positive filtering | Rejects non-date numeric fragments (currency, ranges like "1-2") that generic date parsers misread as dates |
| Project clustering | Groups reworded messages about the same project together via TF-IDF + cosine similarity, tested across multiple phrasings of the same topic |

_(Formal precision/recall benchmarking against a labelled message set is listed under Future Improvements.)_

### Project Structure

taskminer/
├── backend/
│ ├── main.py # FastAPI app — endpoints + alarm scheduling
│ ├── nlp_core.py # task/project/deadline/priority extraction
│ ├── db.py # SQLAlchemy session/engine setup
│ ├── models.py # ORM models (Project, Task)
│ ├── test_nlp_core.py # standalone test of the NLP core
│ └── requirements.txt
└── frontend/
└── index.html # single-file UI — board, projects view, alarm


### Architecture

flowchart TD

    A["User Input<br/>Free-form Natural Language Text"]

    B["NLP Processing<br/>TaskMiner NLP Core"]

    C["Deadline Extraction<br/>• dateparser<br/>• Relative dates<br/>• Weekday rules<br/>• Custom regex filtering"]

    D["Priority Detection<br/>• Urgency keywords<br/>• Time-to-deadline"]

    E["Project Matching<br/>TF-IDF + Cosine Similarity<br/>Against Existing Projects"]

    F{"Matching Project<br/>Found?"}

    G["Existing Project<br/>Assign Task"]

    H["Create New Project<br/>Store Project Corpus"]

    I["Task + Project + Deadline + Priority<br/>Persist in SQLite<br/>SQLAlchemy"]

    J["Alarm Scheduling<br/>APScheduler<br/>Deadline − 15 Minutes"]

    K["Alarm Triggered<br/>Set Active Alarm Flag"]

    L["Frontend Polling<br/>Vanilla HTML / CSS / JS"]

    M["Persistent Alarm Popup<br/>• Looping Alert<br/>• Mark Done<br/>• Dismiss"]

    N{"User Action"}

    O["Mark Task Done<br/>Move To-Do → Done"]

    P["Dismiss Alarm<br/>Alarm Stops"]

    Q["To-Do / Done Board<br/>Real-time Task Updates"]

    R["Projects View<br/>• Tasks by Project<br/>• Progress Tracking"]

    A --> B

    B --> C
    B --> D
    B --> E

    E --> F
    F -->|Yes| G
    F -->|No| H

    C --> I
    D --> I
    G --> I
    H --> I

    I --> J
    J --> K
    K --> L
    L --> M

    M --> N
    N -->|Mark Done| O
    N -->|Dismiss| P

    O --> Q
    P --> Q

    I --> Q
    I --> R

    style A fill:#E8F4FD,stroke:#1F4E79,stroke-width:2px
    style B fill:#FFF2CC,stroke:#BF9000,stroke-width:2px
    style C fill:#FCE5CD,stroke:#E69138,stroke-width:2px
    style D fill:#FCE5CD,stroke:#E69138,stroke-width:2px
    style E fill:#D9EAD3,stroke:#38761D,stroke-width:2px
    style F fill:#D0E0E3,stroke:#134F5C,stroke-width:2px
    style G fill:#D9EAD3,stroke:#38761D,stroke-width:2px
    style H fill:#D9EAD3,stroke:#38761D,stroke-width:2px
    style I fill:#D9D2E9,stroke:#674EA7,stroke-width:2px
    style J fill:#D0E0E3,stroke:#134F5C,stroke-width:2px
    style K fill:#F4CCCC,stroke:#CC0000,stroke-width:2px
    style L fill:#CFE2F3,stroke:#0B5394,stroke-width:2px
    style M fill:#F4CCCC,stroke:#CC0000,stroke-width:3px
    style N fill:#D0E0E3,stroke:#134F5C,stroke-width:2px
    style O fill:#D9EAD3,stroke:#38761D,stroke-width:2px
    style P fill:#FFF2CC,stroke:#BF9000,stroke-width:2px
    style Q fill:#CFE2F3,stroke:#0B5394,stroke-width:2px
    style R fill:#CFE2F3,stroke:#0B5394,stroke-width:2px
### Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Language | Python | Standard for NLP/backend work |
| API | FastAPI | Async, integrates directly with the Python NLP stack |
| Deadline parsing | dateparser + custom regex rules | Resolves absolute/relative dates; custom rules fix known mis-parses (see Limitations) |
| Project clustering | scikit-learn (TF-IDF + cosine similarity) | Matches new text against each project's accumulated corpus, fully offline |
| Database | SQLite via SQLAlchemy | Simple local persistence; swappable for PostgreSQL |
| Scheduling | APScheduler | Triggers the pre-deadline alarm in the background |
| Frontend | Vanilla HTML/CSS/JS | Fast, no build step, real-time board/alarm updates |

### How It Works

1. **Task Input** — user types a task in plain language
2. **NLP Processing**
   - Deadline extraction (dateparser + weekday/relative-date rules)
   - Priority heuristic (urgency keywords + time-to-deadline)
   - Project matching (TF-IDF similarity against existing projects; new project created if no match clears the threshold)
3. **Persistence** — task + project saved to SQLite
4. **Alarm Scheduling** — APScheduler computes (deadline − 15 min) and flips the task's alarm flag at that time
5. **Frontend Polling** — the UI polls for active alarms and shows a persistent, looping alert until the user dismisses it or marks the task done
6. **Board Update** — completed tasks move from To-Do to Done in real time

### Running Locally
