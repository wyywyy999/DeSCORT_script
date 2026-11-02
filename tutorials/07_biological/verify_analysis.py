# Support both an installed package and direct execution from this checkout.
from pathlib import Path as _Path
import sys as _sys
_SOURCE = _Path(__file__).resolve().parents[2] / "src"
if str(_SOURCE) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE))

from pathlib import Path
import importlib.util,json
import pandas as pd,numpy as np
r=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('analysis',r/'01_umap_analysis.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
a=m.parser().parse_args([]);checks={}
for rep in ['model','original']:
 files=m.get_files(getattr(a,rep+'_input'),a.plates)
 features,_=m.inspect_input(files,rep,a);features=features[:4]
 actual=m.aggregate(files[:1],features,a).sort_index()
 raw=pd.read_parquet(files[0],columns=[a.gene_column,a.guide_column]+features)
 reference=raw.groupby([a.gene_column,a.guide_column],sort=False)[features].mean().astype('float32').groupby(level=a.gene_column,sort=False).median().astype('float32')
 reference.index=reference.index.astype(str);reference=reference.loc[reference.index!=a.control_key_word].sort_index()
 pd.testing.assert_index_equal(actual.index,reference.index)
 np.testing.assert_allclose(actual,reference,rtol=2e-5,atol=2e-5)
 checks[rep+'_raw_aggregation']=dict(file=str(files[0]),raw_rows=len(raw),features=features,genes=len(actual),matches_pandas=True)
old=r.parent/'07_biological/corum_string_umap/A549_GO_KEGG_ANNOTATED_pathway_UMAP/results/pathway_distance_df_model.csv'
new=r/'result/01_umap_analysis/model/results/pathway_distance_df_model.csv'
x=pd.read_csv(old).sort_values(['database','pathway']).reset_index(drop=True);y=pd.read_csv(new).sort_values(['database','pathway']).reset_index(drop=True)
assert len(x)==len(y)==998
pd.testing.assert_frame_equal(x[['database','pathway','n_genes','genes']],y[['database','pathway','n_genes','genes']])
error=float(np.max(np.abs(x.mean_distance-y.mean_distance)));np.testing.assert_allclose(x.mean_distance,y.mean_distance,rtol=2e-5,atol=2e-5)
checks['model_pathway_distance_replay']=dict(pathways=len(x),max_absolute_error=error,all_match=True)
json.dump(checks,open(r/'result/01_umap_analysis/verification.json','w'),indent=2)
print(json.dumps(checks,indent=2))
