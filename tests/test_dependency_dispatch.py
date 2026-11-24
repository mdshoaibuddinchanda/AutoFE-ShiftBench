from src.task_manifest import ManifestStore,build_task_records


def test_claim_cannot_bypass_unready_dependency(tmp_path):
    records=build_task_records(['d'],[42],[1,2],[('clean',0)],['Raw'],['logistic_regression'])
    store=ManifestStore(tmp_path/'m.db')
    store.create_run('r',{'execution':{'max_attempts':2}},records)
    first,model=records[:2]
    assert store.claim_task('r',model['scientific_task_id']) is None
    attempt=store.claim_task('r',first['scientific_task_id'])
    store.commit_result('r',first['scientific_task_id'],attempt,{'status':'completed'})
    assert store.claim_task('r',model['scientific_task_id']) is not None
    # Unrelated fold two remains unprepared without blocking fold one.
    assert store.get_task('r',records[2]['scientific_task_id'])['state'] == 'pending'


def test_no_precompute_means_no_phantom_dependency(tmp_path):
    record=build_task_records(['d'],[42],[1],[('clean',0)],['Raw'],['logistic_regression'],include_precompute=False)[0]
    assert record['depends_on'] == []
