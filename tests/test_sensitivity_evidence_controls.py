import json
from src.sensitivity_analysis import analyze_sensitivity
from tests.test_prefix_information import prefix_fixture,config


def test_identical_export_duplicates_are_not_payload_conflicts(tmp_path):
    store,_,rows,ledger=prefix_fixture(tmp_path)
    with ledger.open('a') as handle:
        handle.write(json.dumps(rows[0])+'\n')
    bundle=analyze_sensitivity(store.db_path,ledger,'r',config=config())
    assert 'conflicting_ledger_rows' not in set(bundle.exclusions.get('reason',[]))
    assert (bundle.coverage_tasks['classification'] == 'completed_valid').all()


def test_latest_attempt_is_sequence_not_hashed_id(tmp_path):
    store,records,_,ledger=prefix_fixture(tmp_path)
    tid=records[0]['scientific_task_id']
    with store._connect() as connection:
        columns=[row[1] for row in connection.execute('PRAGMA table_info(attempts)')]
        row=dict(connection.execute('SELECT * FROM attempts WHERE scientific_task_id=?',(tid,)).fetchone())
        connection.execute('UPDATE attempts SET attempt_id=? WHERE scientific_task_id=?',('zz_old',tid))
        row.update(attempt_id='aa_new',attempt_number=2,state='failed',started_at='2026-10-03T12:00:00+00:00')
        connection.execute('INSERT INTO attempts ('+','.join(columns)+') VALUES ('+','.join('?' for _ in columns)+')',tuple(row[key] for key in columns))
    bundle=analyze_sensitivity(store.db_path,ledger,'r',config=config())
    task=bundle.coverage_tasks[bundle.coverage_tasks.scientific_task_id == tid].iloc[0]
    assert task.latest_attempt_id == 'aa_new'
