import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from src.preprocessing import _build_preprocessor,_to_dense_array,processed_frame
from src.feature_engineering import expand_features_with_dfs,DFSConfig
from src.resource_limits import ResourceLimitError


def test_exact_high_cardinality_encoding_is_sparse_and_budgeted():
    frame=pd.DataFrame({'category':['category_'+str(i) for i in range(2000)]})
    pre=_build_preprocessor(frame,'onehot',False)
    values=pre.fit_transform(frame)
    assert sparse.issparse(values)
    assert values.shape == (2000,2000)
    assert values.nnz == 2000
    assert values.dtype == np.float64
    assert len(pre.named_transformers_['cat'].named_steps['encoder'].categories_[0]) == 2000
    with pytest.raises(ResourceLimitError):
        _to_dense_array(values,budget=1_000_000)


def test_sparse_encoding_preserves_dense_reference_values_selections_and_history():
    frame=pd.DataFrame({'a':[1.,2.,3.,4.,5.,6.],'kind':['a','b','c','a','b','c']})
    pre=_build_preprocessor(frame,'onehot',True)
    values=pre.fit_transform(frame)
    sparse_frame=processed_frame(values,pre.get_feature_names_out())
    dense_frame=pd.DataFrame(_to_dense_array(values),columns=pre.get_feature_names_out())
    config=DFSConfig(max_base_features=2,selection_method='variance')
    a=expand_features_with_dfs(sparse_frame,sparse_frame,np.tile([0,1],3),config)
    b=expand_features_with_dfs(dense_frame,dense_frame,np.tile([0,1],3),config)
    pd.testing.assert_frame_equal(a[0],b[0])
    assert a[2]['selection_history'] == b[2]['selection_history']
