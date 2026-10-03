"""Reconstruct task/result visibility at a transactional logical event cutoff."""
from __future__ import annotations

class HistoricalSnapshotUnsupported(ValueError):
    pass


def completed_model_events(snapshot):
    tasks={row["scientific_task_id"]:row for row in snapshot["tasks"]}
    durable={(row["scientific_task_id"],row["attempt_id"]):row for row in snapshot["durable_results"]+snapshot.get('superseded_results',[])}
    events=snapshot.get("task_events",[])
    first={}
    for event in events:
        first.setdefault(event["scientific_task_id"],event)
    created={key for key,event in first.items() if event["attempt_id"] is None
        and event["state"] == json_payload(tasks[key]).get("initial_state","pending")}
    if snapshot["run"].get("schema_version") != "task_manifest_logical_history_v2" or not set(tasks).issubset(created):
        raise HistoricalSnapshotUnsupported("No complete task state history; wall timestamps alone do not establish historical snapshots")
    return [event for event in events if event["state"] == "completed"
        and tasks[event["scientific_task_id"]]["stage"] == "model"
        and (event["scientific_task_id"],event["attempt_id"]) in durable]


def json_payload(task):
    import json
    return json.loads(task["payload_json"])


def at_event(snapshot,cutoff_order):
    completed_model_events(snapshot)
    events=[row for row in snapshot["task_events"] if row["event_order"] <= cutoff_order]
    last={row["scientific_task_id"]:row for row in events}
    claims={row["attempt_id"]:row for row in events if row["state"] == "running"}
    terminal={row["attempt_id"]:row for row in events if row["attempt_id"] and row["state"] != "running"}
    durable=[row for row in snapshot["durable_results"]+snapshot.get('superseded_results',[]) if row["attempt_id"] in terminal
        and terminal[row["attempt_id"]]["state"] == "completed" and last[row['scientific_task_id']]['attempt_id'] == row['attempt_id']]
    durable_by_task={row["scientific_task_id"]:row for row in durable}
    tasks=[]
    for current in snapshot["tasks"]:
        event=last.get(current["scientific_task_id"])
        if event is None:
            raise HistoricalSnapshotUnsupported("Task declaration was not visible at cutoff")
        task=dict(current)
        task.update(state=event["state"],active_attempt_id=event["attempt_id"] if event["state"] == "running" else None,
            outcome_reason=event["outcome_reason"],updated_at=event["recorded_at"],
            attempt_count=sum(row["scientific_task_id"] == task["scientific_task_id"] for row in claims.values()),
            result_ref=durable_by_task.get(task["scientific_task_id"],{}).get("result_ref"))
        if task["state"] == "completed" and task["scientific_task_id"] not in durable_by_task:
            raise HistoricalSnapshotUnsupported("Historical completed result payload no longer available")
        tasks.append(task)
    attempts=[]
    for current in snapshot["attempts"]:
        if current["attempt_id"] not in claims:
            continue
        attempt=dict(current)
        event=terminal.get(attempt["attempt_id"])
        if event is None:
            attempt.update(state="running",finished_at=None,elapsed_seconds=None,outcome=None,exception_type=None,exception_message=None,traceback=None,result_ref=None)
        else:
            attempt["state"]="completed" if event["state"] == "completed" else ("timeout" if event["outcome_reason"] == "timeout" else "failed")
        attempts.append(attempt)
    run={key:snapshot["run"][key] for key in ("run_id","config_json","protocol_version","seed_scheme_version","schema_version")}
    return {"run":run,"tasks":tasks,"attempts":attempts,"durable_results":durable,"task_events":events,
        "cutoff_event_order":int(cutoff_order),"historical_scope":"transactional_logical_task_state_and_result_visibility"}
