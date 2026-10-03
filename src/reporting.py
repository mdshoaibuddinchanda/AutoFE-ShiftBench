"""One declared run/scope and dataset analysis contract for active reports."""
from dataclasses import replace
import json
from pathlib import Path
import pandas as pd
from src.dataset_statistics import AnalysisConfig,AnalysisInputError,analyze_ledger,_read_ledger,_fingerprint,_validate_protocol,_validate_pipeline_semantics,_finite_metric

REPORT_SCHEMA='dataset_unit_reports_v2'


def report_inputs(ledger,*,config=None,datasets=None,output_dir=None):
    config=config or AnalysisConfig()
    if datasets is not None:
        config=replace(config,datasets=tuple(sorted(datasets)))
    path=Path(ledger)
    rows,exclusions,input_fingerprint=_read_ledger(path,return_fingerprint=True)
    bundle=analyze_ledger(ledger,config,output_dir=output_dir,_parsed_input=(rows,exclusions,input_fingerprint))
    config=replace(config,run_id=bundle.config['run_id'])
    selected=[row for row in rows if row.get('run_id') == config.run_id and (config.datasets is None or row.get('dataset') in config.datasets)]
    for pipeline in {row.get('pipeline') for row in selected}:
        declared=replace(config,pipeline_a=pipeline,pipeline_b=pipeline)
        _validate_protocol(selected,declared)
        _validate_pipeline_semantics(selected,declared)
    valid=[row for row in selected if row.get('status') == 'success' and _finite_metric(row,config.metric)[0]]
    frame=pd.DataFrame(valid)
    frame.attrs={'report_schema':REPORT_SCHEMA,'analysis_config':bundle.config,
        'corrected_summaries':json.loads(bundle.summaries.to_json(orient='records')),
        'exclusion_count':len(bundle.exclusions),'scope':'observed paired dataset contrasts; missing work is not imputed'}
    return frame,bundle


def dataset_descriptive(frame,groups,metrics):
    """Each dataset contributes one mean; descriptive only, without pooled CI."""
    if frame.empty: return frame
    available=[metric for metric in metrics if metric in frame]
    return frame.groupby(['dataset',*groups],dropna=False)[available].mean().reset_index()


def summaries(frame):
    if frame.attrs.get('report_schema') != REPORT_SCHEMA:
        raise AnalysisInputError('Inferential figures require corrected report_inputs and its declared bundle')
    return pd.DataFrame(frame.attrs['corrected_summaries'])


def caption(bundle):
    supported=bundle.summaries[bundle.summaries['status'].isin(['complete','all_zero_effect'])] if not bundle.summaries.empty else bundle.summaries
    return f"Run {bundle.config['run_id']}; {len(bundle.dataset_contrasts)} dataset/stratum contrasts; {len(supported)} supported stratum tests; {len(bundle.exclusions)} exclusions. Effects are {bundle.config['pipeline_b']} minus {bundle.config['pipeline_a']}; intervals resample datasets and tests use declared Holm families. Missing/unsupported outcomes are explicit."
