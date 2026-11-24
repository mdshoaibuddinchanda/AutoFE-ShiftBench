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


def test_canonical_manuscript_figure_export_contract():
    from reproduction import figures

    assert figures.FIGURE_DPI == 350
    assert figures.DEFAULT_OUTPUT.name == 'figures'
    assert figures.SUBMISSION_OUTPUT.name == 'KNOSYS-D-26-17927'
