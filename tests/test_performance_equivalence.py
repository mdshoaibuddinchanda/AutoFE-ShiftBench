import hashlib
import numpy as np
import pandas as pd
import pytest
from src import feature_engineering as fe
from src.operator_registry import candidate_id,OPERATOR_SET_REGISTRY


@pytest.mark.parametrize('selector',['variance','random','none','mi'])
@pytest.mark.parametrize('operators',list(OPERATOR_SET_REGISTRY))
def test_streaming_and_train_only_selection_match_full_matrix(selector,operators):
    rng=np.random.default_rng(926)
    train=pd.DataFrame({'a':rng.normal(size=36),'b':np.tile([0.,1.,1.],12),'c':np.zeros(36)})
    train.attrs['discrete_features']=['b','c']
    test=pd.DataFrame({'a':[1e9,0.,-1e9],'b':[0.,1.,1.],'c':[0.,0.,0.]},index=[90,91,92])
    cfg=fe.DFSConfig(operator_set_id=operators,selection_method=selector,max_features=5,monitor_ram=False,random_seed=777)
    numeric,bases,_=fe._base_columns(train,cfg)
    expressions=fe._generate_expressions(bases,operators,cfg.depth)
    values={candidate_id(expr):fe._evaluate_expression(expr,numeric)[0] for expr in expressions}
    matrix=pd.DataFrame(values,index=numeric.index)
    matrix.attrs['discrete_features']=[candidate_id(expr) for expr in expressions if set(fe.expression_columns(expr)).issubset({'b','c'})]
    scores=fe._score_candidates(matrix,np.arange(36)%2,cfg)
    ranked=sorted(matrix,key=lambda fid:(-(scores[fid] if scores[fid] is not None and np.isfinite(scores[fid]) else -np.inf),fid))
    selected=ranked[:5]
    expected_test=pd.DataFrame({candidate_id(expr):fe._evaluate_expression(expr,fe._numeric_frame(test))[0] for expr in expressions},index=test.index)[selected]
    actual_train,actual_test,meta=fe.expand_features_with_dfs(train,test,np.arange(36)%2,cfg)
    pd.testing.assert_frame_equal(actual_train,matrix[selected],check_exact=True)
    pd.testing.assert_frame_equal(actual_test,expected_test,check_exact=True)
    assert meta['selected_feature_identities']==selected
    assert meta['candidate_count']==len(expressions)
    for event in meta['selection_history']:
        assert event['score']==scores[event['candidate_id']]
        assert event['candidate_rank']==ranked.index(event['candidate_id'])+1


def test_only_requested_pipeline_is_generated_and_dependency_is_unchanged(tmp_path,monkeypatch):
    from src import pipeline_runner as runner
    monkeypatch.chdir(tmp_path)
    x=pd.DataFrame({'x':[1.,2.,3.,4.],'z':[3.,2.,1.,4.]})
    first=runner._run_pipeline_generation(x,x,np.array([0,1,0,1]),'d','stratified',42,1,'clean',selected_pipelines=['Raw'])
    assert set(first[0])=={'Raw'}
    assert len(list(tmp_path.rglob('*_train.pkl')))==1
    second=runner._run_pipeline_generation(x,x,np.array([0,1,0,1]),'d','stratified',42,1,'clean')
    assert first[1]['Raw']['dependency_signature']==second[1]['Raw']['dependency_signature']
    pd.testing.assert_frame_equal(first[0]['Raw'][0],second[0]['Raw'][0],check_exact=True)


def test_report_parses_once_and_fingerprints_exact_parsed_bytes(tmp_path,monkeypatch):
    from src import reporting,dataset_statistics as ds
    from tests.test_dataset_statistics import _record
    import json
    path=tmp_path/'ledger.jsonl'
    path.write_bytes(('\r\n'.join(json.dumps(_record(d,42,p,.7)) for d in ('a','b') for p in ('Raw','AutoFE_Baseline'))+'\r\n').encode())
    original=reporting._read_ledger
    calls=[]
    def counted(*a,**kw): calls.append(1);return original(*a,**kw)
    monkeypatch.setattr(reporting,'_read_ledger',counted)
    monkeypatch.setattr(ds,'_read_ledger',lambda *a,**kw:pytest.fail('second parse'))
    frame,bundle=reporting.report_inputs(path)
    assert calls==[1]
    assert bundle.input_fingerprint==hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('x,y',[(np.array([1,2,2,3]),np.array([1,1,4])),(np.array([]),np.array([1.])),(np.array([np.nan]),np.array([0.])),(np.array([np.inf]),np.array([np.inf])),(np.array([-np.inf,1]),np.array([np.inf,2])),(np.array([1e308,-1e308]),np.array([0.,0.]))])
def test_sorted_cliff_delta_preserves_pairwise_definition(x,y):
    from src.stats_analysis import cliffs_delta
    with np.errstate(all='ignore'):
        expected=float(np.sign(np.asarray(x,dtype=float)[:,None]-np.asarray(y,dtype=float)[None,:]).mean()) if len(x) and len(y) else np.nan
    actual=cliffs_delta(x,y)
    assert actual==expected or np.isnan(actual) and np.isnan(expected)
