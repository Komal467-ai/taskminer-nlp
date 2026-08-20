from nlp_core import TaskMinerNLP

engine = TaskMinerNLP()

# Simulating your example: data entered across 2 days about the same project,
# then a message that should be clearly identified as its own project.
messages = [
    "Need to finalize the SecureBERT fine-tuning results for the cyber threat intel project",
    "Cyber threat intel: write up the precision/recall numbers, due tomorrow 6pm",
    "For the threat classification project, prepare the streamlit dashboard slides by Friday 5pm urgent",
    "Buy groceries for the week, day after tomorrow",
    "Threat intel project - fix the SecureBERT tokenizer bug before submission deadline next Monday 9am",
]

print("=" * 70)
for msg in messages:
    task = engine.process_message(msg)
    proj = engine.projects[task.project_id]
    print(f"MSG: {msg}")
    print(f"  -> project   : {proj.name}  (id={proj.id})")
    print(f"  -> task      : {task.description}")
    print(f"  -> deadline  : {task.deadline}")
    print(f"  -> priority  : {task.priority}")
    print("-" * 70)

print("\nProjects formed:", len(engine.projects))
for p in engine.projects.values():
    print(f"  [{p.id}] {p.name} -> {len(p.task_ids)} task(s)")