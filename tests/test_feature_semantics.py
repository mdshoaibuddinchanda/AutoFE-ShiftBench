import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import DFSConfig,_generate_expressions,expand_features_with_dfs
from src.operator_registry import candidate_id
from src.fsva import selection_stability
from src.shift_generator import apply_perturbation


def test_depth_two_has_unique_candidates():
    exprs=_generate_expressions(['a','b'],'full_arithmetic_v1',2)
    assert len(exprs) == len({candidate_id(expr) for expr in exprs})


def test_zero_removal_preserves_exact_frame():
    x=pd.DataFrame({'a':[1,2],'b':[3,4]})
    got,_=apply_perturbation(x,pd.Series([0,1]),'feature_removal',0,42)
    pd.testing.assert_frame_equal(got,x)


@pytest.mark.parametrize('field',['max_features','max_base_features'])
def test_nonpositive_caps_rejected(field):
    with pytest.raises(ValueError):
        DFSConfig(**{field:0})


def test_stability_reads_authoritative_selection_history():
    history={'selected_feature_identities':['a'],'selection_history':[{'candidate_id':'a','eligible':True,'selected':True}]}
    got=selection_stability([history,history])
    assert got['candidate_availability'] == {'a':2}
    assert got['candidate_selection_frequency'] == {'a':2}
    with pytest.raises(ValueError):
        selection_stability([{**history,'candidate_history':[]}])


def test_mi_uses_declared_discrete_provenance(monkeypatch):
    x=pd.DataFrame({'integer_measurement':[0,1,2,3], 'one_hot':[0,1,0,1]})
    x.attrs['discrete_features']=['one_hot']
    seen=[]
    def mi(frame,y,**kwargs):
        seen.extend(kwargs.get('discrete_features',[]))
        return np.zeros(frame.shape[1])
    monkeypatch.setattr('src.feature_engineering.mutual_info_classif',mi)
    _,_,meta=expand_features_with_dfs(x,x,np.array([0,1,0,1]),DFSConfig(enable_dfs=False,operator_set_id='none_v1',selection_method='mi'))
    assert seen == [False,True]
    assert meta['mi_semantics_version'] == 'declared_discrete_provenance_v2'


def test_runner_counts_and_training_distance_are_real(tmp_path,monkeypatch):
    from src import pipeline_runner as runner
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner,'PIPELINE_CONFIGS',{'AutoFE_Baseline':runner.PIPELINE_CONFIGS['AutoFE_Baseline']})
    rng=np.random.default_rng(61)
    frame=pd.DataFrame({'a':rng.normal(size=60),'b':rng.normal(size=60),'target':np.tile(['a','b'],30)})
    frame.to_csv('d.csv',index=False)
    clean=runner.get_data_splits('d.csv','d',42,1,'clean','clean',0)
    corrupt=runner.get_data_splits('d.csv','d',42,1,'gaussian_noise_0.1','gaussian_noise',.1)
    meta=corrupt[4]['AutoFE_Baseline']
    assert meta['num_generated'] > 0
    assert meta['num_generated']+meta['num_original'] == meta['candidate_count']
    assert meta['held_out_distribution_distance']['wasserstein'] == 0
    assert meta['training_distribution_distance']['wasserstein'] > 0
    assert clean[4]['AutoFE_Baseline']['training_distribution_distance']['wasserstein'] == 0


def test_historical_relabel_preserves_inputs_but_changes_conditionals():
    from src.pipeline_runner import apply_training_condition
    from src.shift_generator import CONDITION_IDENTITIES
    x=pd.DataFrame({'x':np.arange(100)})
    y=pd.Series([0]*80+[1]*20)
    a,b,heldout,labels=apply_training_condition(x,y,x_test=x,y_test=y,shift_family='class_prior_shift',severity=0,random_state=42)
    pd.testing.assert_frame_equal(a,x)
    pd.testing.assert_frame_equal(heldout,x)
    pd.testing.assert_series_equal(labels,y)
    assert b.value_counts().to_dict() == {1:60,0:40}
    assert a.loc[b == 1,'x'].min() < 80
    assert CONDITION_IDENTITIES['class_prior_shift'] == 'majority_half_relabel_training_v1'
    again=apply_training_condition(x,y,x_test=x,y_test=y,shift_family='class_prior_shift',severity=0,random_state=42)
    pd.testing.assert_series_equal(b,again[1])
