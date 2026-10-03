import json
from pathlib import Path
import pandas as pd
from src.reporting import report_inputs
from src.dataset_statistics import AnalysisConfig
from src.generate_tables import generate_table_2_robustness,generate_all_tables
from tests.test_dataset_statistics import _record


def fixture(tmp_path):
    rows=[_record(d,1,p,v) for d in ('d1','d2','d3') for p,v in (('Raw',.6),('AutoFE_Baseline',.7))]
    path=tmp_path/'r.jsonl'
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    return path


def test_missing_table_conditions_are_missing_not_zero():
    frame=pd.DataFrame({'pipeline':['Raw'],'condition':['clean'],'dataset':['d'],'roc_auc':[.7]})
    text=generate_table_2_robustness(frame)
    assert 'NA (unavailable)' in text
    assert '0.0000' not in text
    assert 'missing_values_0.1' in text


def test_reporting_scope_and_resampling_flags_use_the_same_bundle(tmp_path):
    path=fixture(tmp_path)
    config=AnalysisConfig(bootstrap_resamples=0,permutation_resamples=0)
    frame,bundle=report_inputs(path,config=config,datasets=['d1','d2'])
    assert set(frame['dataset']) == {'d1','d2'}
    assert set(bundle.dataset_contrasts['dataset']) == {'d1','d2'}
    assert bundle.config['bootstrap_resamples'] == 0
    assert bundle.summaries['ci_lower'].isna().all()
    assert bundle.summaries['p_value'].isna().all()
    output=generate_all_tables(path,analysis_config=config,datasets=['d1','d2'],output_dir=tmp_path/'tables')
    text=output.read_text()
    assert 'Legacy Exploratory Pooled' not in text
    assert 'Holm' in text


def test_active_figure_uses_corrected_contrasts_without_row_bootstrap(tmp_path,monkeypatch):
    from src import plotting_q1 as plotting
    frame,bundle=report_inputs(fixture(tmp_path),config=AnalysisConfig(bootstrap_resamples=100,permutation_resamples=100))
    monkeypatch.setattr(plotting.sns,'barplot',lambda *a,**k:(_ for _ in ()).throw(AssertionError('pooled row bar bootstrap')))
    saved=[]
    monkeypatch.setattr(plotting,'_save_figure',lambda fig,out,stem,dpi:saved.append((stem,dpi)))
    plotting.plot_fig3_pipeline_ranking(frame,tmp_path)
    assert saved == [('Figure_3_Pipeline_Ranking',300)]
