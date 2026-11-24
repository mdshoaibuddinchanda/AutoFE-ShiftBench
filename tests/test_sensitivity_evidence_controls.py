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


def test_indexed_selection_matches_row_oracle_without_changing_order(tmp_path):
    import numpy as np
    import pandas as pd
    from src.sensitivity_analysis import _pair_keys,_selection_mask,_normalise_pairs_for_analysis
    store,_,rows,ledger=prefix_fixture(tmp_path)
    pairs=analyze_sensitivity(store.db_path,ledger,'r',config=config()).pair_blocks
    keys=_pair_keys(pairs)
    selected=set(keys[::2])
    oracle=pairs.apply(lambda row:(str(row['contrast_id']),str(row['stratum']),tuple(row['task_key_tuple'])) in selected,axis=1).to_numpy()
    mask=_selection_mask(keys,selected)
    np.testing.assert_array_equal(mask,oracle)
    expected=pairs.copy();expected['selected']=oracle;expected['regime']='probe'
    pd.testing.assert_frame_equal(_normalise_pairs_for_analysis(pairs,'probe',selected,_selected_mask=mask),expected,check_exact=True)
