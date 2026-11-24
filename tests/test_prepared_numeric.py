from pathlib import Path
import gc
import numpy as np
import pandas as pd
import pytest
from src import pipeline_runner as runner
from src.prepared_inputs import publish_unit,load_pipeline,verify_unit,historical_model_array
from src.model import build_model


def prepared(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    rng=np.random.default_rng(862)
    frame=pd.DataFrame(rng.normal(size=(90,6)),columns=[f'x{i}' for i in range(6)])
    frame['target']=np.arange(90)%3
    frame.to_csv('d.csv',index=False)
    value=runner.get_data_splits('d.csv','d',42,1,'clean','clean',0,selected_pipelines=['Raw'])
    descriptor=publish_unit(Path('unit.pkl'),value,input_artifacts=value[4]['Raw']['input_artifacts'])
    return value,descriptor


def test_numeric_loader_is_readonly_exact_and_avoids_feature_unpickle(tmp_path,monkeypatch):
    value,descriptor=prepared(tmp_path,monkeypatch)
    original=pd.read_pickle
    calls=[]
    def tracked(path):
        calls.append(str(path))
        assert str(path)=='unit.pkl'
        return original(path)
    monkeypatch.setattr(pd,'read_pickle',tracked)
    loaded=load_pipeline(descriptor,'Raw',model_numeric=True)
    arrays=loaded[0]['Raw']
    for actual,source in zip(arrays,value[0]['Raw']):
        np.testing.assert_array_equal(actual,historical_model_array(source),strict=True)
        assert not actual.flags.writeable
        actual._mmap.close()
    assert calls==['unit.pkl']
    assert loaded[4]['Raw']['model_numeric_policy']


@pytest.mark.parametrize('name',['random_forest','extra_trees','linear_svm','knn','logistic_regression','gaussian_nb','mlp','lightgbm'])
def test_actual_cpu_factory_accepts_readonly_matrices_without_changing_outputs(tmp_path,monkeypatch,name):
    value,descriptor=prepared(tmp_path,monkeypatch)
    loaded=load_pipeline(descriptor,'Raw',model_numeric=True)
    train,test=loaded[0]['Raw']
    dense=historical_model_array(value[0]['Raw'][0])
    y=value[1]
    first=build_model(name,random_state=42,use_gpu=False);second=build_model(name,random_state=42,use_gpu=False)
    first.fit(dense,y);second.fit(train,y)
    np.testing.assert_array_equal(first.predict(historical_model_array(value[0]['Raw'][1])),second.predict(test),strict=True)
    np.testing.assert_array_equal(first.predict_proba(historical_model_array(value[0]['Raw'][1])),second.predict_proba(test),strict=True)
    del second;gc.collect()
    train._mmap.close();test._mmap.close()
    item=next(x for x in loaded[4]['Raw']['executed_artifacts'] if x['role']=='model_train_numeric_matrix')
    source=Path(item['path']);moved=source.with_suffix('.closed')
    source.rename(moved);moved.rename(source)


def test_missing_numeric_artifact_repairs_original_bytes_without_descriptor_rewrite(tmp_path,monkeypatch):
    value,descriptor=prepared(tmp_path,monkeypatch)
    unit=pd.read_pickle('unit.pkl')
    path=Path(unit['model_inputs']['Raw']['train']['path'])
    original=path.read_bytes();path.unlink()
    with pytest.raises(OSError):verify_unit(descriptor)
    assert publish_unit('unit.pkl',value,input_artifacts=value[4]['Raw']['input_artifacts'])==descriptor
    assert path.read_bytes()==original
    verify_unit(descriptor)


def test_array_publication_preserves_signed_zero_nan_and_fortran_layout(tmp_path):
    from src.prepared_inputs import _publish_array
    frame=pd.DataFrame({'x':[-0.,np.nan,np.inf],'y':[0.,-np.inf,1.]})
    values=historical_model_array(frame)
    path=tmp_path/'values.npy';_publish_array(path,values)
    mapped=np.load(path,mmap_mode='r',allow_pickle=False)
    np.testing.assert_array_equal(mapped.view(np.uint32),values.view(np.uint32))
    assert bool(mapped.flags.f_contiguous)==bool(values.flags.f_contiguous)
    with pytest.raises(ValueError):mapped[0,0]=1
    mapped._mmap.close()
    before=path.read_bytes();_publish_array(path,values)
    assert path.read_bytes()==before
