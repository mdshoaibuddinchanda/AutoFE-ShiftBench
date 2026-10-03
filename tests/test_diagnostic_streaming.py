from collections import Counter
import numpy as np
import pandas as pd
import pytest
from src import fsva
from src.operator_registry import raw_expression,op_expression


@pytest.mark.parametrize('extreme',[False,True])
def test_streamed_jacobian_matches_tensor_definition_and_exact_status(extreme):
    rng=np.random.default_rng(731)
    frame=pd.DataFrame(rng.normal(size=(40,80)),columns=[f'x{i}' for i in range(80)])
    frame.x1=np.tile([0.,1e-14,1.,-1.],10)
    if extreme: frame.loc[0,['x0','x2']]=[1e308,1e308]
    a,b,c=map(raw_expression,['x0','x1','x2'])
    exprs=[a,op_expression('add_numeric',a,b),op_expression('multiply_numeric',a,b),op_expression('divide_numeric',a,b),op_expression('multiply_numeric',op_expression('multiply_numeric',a,c),b)]
    sampled=fsva._sample_rows(frame,32,3)
    counts=Counter();jacs=[]
    with np.errstate(all='ignore'):
        for expr in exprs:
            _,jac,status=fsva._derivative(expr,sampled);jacs.append(jac);counts.update(status)
        expected=fsva._summary(np.stack(jacs,axis=1),status=dict(counts))
        actual=fsva.compute_jacobian_diagnostic(frame,exprs,raw_control_expressions=[a,b,c],max_rows=32,random_state=3)
        tensor=fsva._tensor_finite_difference(frame,exprs,max_rows=32,random_state=3)
        streamed=fsva.validate_jacobian_finite_difference(frame,exprs,max_rows=32,random_state=3)
    assert actual['selected']['status']==expected['status']
    for key,value in expected.items():
        if isinstance(value,float): assert np.isclose(actual['selected'][key],value,rtol=1e-12,atol=1e-12,equal_nan=True)
        else: assert actual['selected'][key]==value
    for key in tensor:
        if key=='mean_absolute_error': assert np.isclose(streamed[key],tensor[key],rtol=1e-12,atol=1e-15)
        else: assert streamed[key]==tensor[key]


def test_wide_empirical_values_retain_derivative_validity():
    frame=pd.DataFrame(np.ones((4,80)),columns=[f'x{i}' for i in range(80)])
    frame.x0=[0.,1e308,1e-14,2.]
    frame.x1=[1.,0.,1e-13,0.]
    a,b=map(raw_expression,['x0','x1'])
    exprs=[op_expression('divide_numeric',a,b),op_expression('multiply_numeric',a,b)]
    with np.errstate(all='ignore'):
        actual,status=fsva._evaluate_mapping(exprs,frame,return_status=True)
        pairs=[fsva._derivative(expr,frame) for expr in exprs]
    np.testing.assert_array_equal(actual,np.stack([pair[0] for pair in pairs],axis=1))
    expected=Counter()
    for _,_,counts in pairs:expected.update(counts)
    assert status==dict(expected)


def test_nonfinite_input_uses_original_finite_difference_semantics():
    frame=pd.DataFrame({'x':[np.nan,2.],'z':[1.,np.inf]})
    expr=[raw_expression('x')]
    with np.errstate(all='ignore'):
        a=fsva.validate_jacobian_finite_difference(frame,expr)
        b=fsva._tensor_finite_difference(frame,expr)
    assert a['status']==b['status']
    assert np.isnan(a['max_absolute_error']) and np.isnan(b['max_absolute_error'])


def test_exact_wide_raw_diagnostics_fit_bounded_workspace(monkeypatch):
    monkeypatch.setenv('AUTOFE_DENSE_BUDGET_BYTES',str(8*1024**2))
    frame=pd.DataFrame(np.eye(800),columns=[f'x{i}' for i in range(800)])
    exprs=[raw_expression(column) for column in frame]
    result=fsva.compute_jacobian_diagnostic(frame,exprs,raw_control_expressions=exprs,max_rows=128,random_state=42)
    assert result['selected']['n_outputs']==800
    assert result['selected']['n_inputs']==800
    assert result['selected']['frobenius_norm']==np.sqrt(128*800)
    validation=fsva.validate_jacobian_finite_difference(frame,exprs,max_rows=32,random_state=42)
    assert validation['status']=='validated'
