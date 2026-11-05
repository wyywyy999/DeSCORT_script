"""Shared input handling, aggregation and figure export for biological analyses."""
from pathlib import Path
import argparse
import json
import logging
import pickle
import tempfile
import os

import duckdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT = Path(__file__).resolve().parents[3]
_ORIGINAL = PROJECT.parents[1] if len(PROJECT.parents) > 1 else PROJECT
_LOCAL_LAYOUT = (_ORIGINAL / "src/descort").is_dir()
WORKSPACE = Path(os.environ.get("DESCORT_WORKSPACE_ROOT", str(_ORIGINAL.parent if _LOCAL_LAYOUT else PROJECT.parent)))
RESOURCE_ROOT = Path(os.environ.get("DESCORT_RESOURCE_ROOT", str(_ORIGINAL / "common_files" if _LOCAL_LAYOUT else PROJECT / "resources")))
TUTORIAL_DIR = PROJECT / 'tutorials/07_biological' 
LOG = logging.getLogger('biological.utils')


def input_defaults():
    """Return a fresh namespace of shared CLI defaults (no argument parsing)."""
    return argparse.Namespace(
        model_input=WORKSPACE/'Paper_LLm/tutorials/01_mAP/well_level_new_site_pretrain_emb_attention_mask/cpg0021_A549_DeSCORT_1382_nosampling_layernorm_finetune_A549/20200805_A549_WG_Screen/normalization_clip_data',
        original_input=Path('/fse/DC/CP_data/feature_data/20200805_A549_WG_Screen/raw_data/well_level_new/normalization_clip_data'),
        original_features_pkl=WORKSPACE/'Final_well_level_new_LLM/common_files/00_pretrain/cpg0021_A549_union_feature_1382.pkl',
        model_feature_prefix='emb_',
        model_feature_dim=256,
        gene_column='Metadata_Foci_Barcode_MatchedTo_GeneCode',
        guide_column='Metadata_Foci_Barcode_MatchedTo_Barcode',
        control_key_word='nontargeting',
        hit_column='Gene',
        threads=8,
        memory_limit='24GB',
        plates=['CP186A','CP186B','CP186C','CP186D','CP186E','CP186F','CP186G','CP186H','CP186N'],
    )


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + '\n')


def get_files(folder, plates):
    files = sorted(folder.glob('*.parquet')) if folder.is_dir() else [folder]
    if plates:
        files = [f for f in files if any(p in f.name for p in plates)]
        if len(files) != len(plates):
            raise ValueError(f'{folder}: expected {len(plates)} plate files, found {len(files)}')
    if not files or not all(f.is_file() for f in files):
        raise FileNotFoundError(f'No input parquet files: {folder}')
    return files


def load_features(path):
    with path.open('rb') as f:
        obj = pickle.load(f)
    if isinstance(obj, pd.DataFrame):
        obj = obj.iloc[:,0].tolist() if obj.shape[1] == 1 else obj.columns.tolist()
    elif isinstance(obj, dict):
        key = next((k for k in ['features','feature','feature_list','need_feature_list','columns'] if k in obj), None)
        if key:
            obj = obj[key]
        else:
            lists = [v for v in obj.values() if isinstance(v,(list,tuple,set,np.ndarray,pd.Index,pd.Series))]
            if len(lists) != 1:
                raise ValueError(f'Cannot determine feature list in {path}')
            obj = lists[0]
    return list(dict.fromkeys(str(v).strip() for v in obj if v is not None and str(v).strip()))


def inspect_input(files, representation, args, result=None):
    required = {args.gene_column, args.guide_column}
    numeric = None
    for f in files:
        schema = pq.ParquetFile(f).schema_arrow
        names = set(schema.names)
        if not required <= names:
            raise ValueError(f'{f}: missing metadata {required-names}')
        numbers = {v.name for v in schema if pa.types.is_integer(v.type) or pa.types.is_floating(v.type) or pa.types.is_decimal(v.type)}
        numeric = numbers if numeric is None else numeric & numbers
    requested = ([f'{args.model_feature_prefix}{i}' for i in range(args.model_feature_dim)]
                 if representation == 'model' else load_features(args.original_features_pkl))
    features = [f for f in requested if f in numeric and not f.startswith('Metadata_') and f not in required]
    excluded = [f for f in requested if f not in features]
    if representation == 'model' and excluded:
        raise ValueError(f'Model features missing or non-numeric: {excluded}')
    if not features:
        raise ValueError('No numeric features available')
    if result:
        pd.DataFrame({'feature':features}).to_csv(result/'features_used.csv',index=False)
        pd.DataFrame({'excluded_feature':excluded}).to_csv(result/'excluded_requested_features.csv',index=False)
    info = dict(files=[str(f) for f in files], rows=sum(pq.ParquetFile(f).metadata.num_rows for f in files),
                requested_features=len(requested), used_features=len(features), excluded_features=excluded)
    LOG.info('%s input: %s', representation, info)
    return features, info


def q(s):
    return '"' + str(s).replace('"','""') + '"'


def aggregate(files, features, args):
    sources = ','.join("'"+str(f).replace("'","''")+"'" for f in files)
    avg = ','.join(f'AVG({q(f)})::FLOAT AS {q(f)}' for f in features)
    sql = f'SELECT {q(args.gene_column)}, {q(args.guide_column)}, {avg} FROM read_parquet([{sources}], union_by_name=true) GROUP BY {q(args.gene_column)}, {q(args.guide_column)}'
    with tempfile.TemporaryDirectory(prefix='descort_pathway_') as temp:
        con = duckdb.connect()
        try:
            con.execute(f'SET threads TO {args.threads}')
            con.execute("SET memory_limit = '"+args.memory_limit.replace("'","''")+"'")
            con.execute("SET temp_directory = '"+temp.replace("'","''")+"'")
            guide = con.execute(sql).fetch_df().set_index([args.gene_column,args.guide_column])
        finally:
            con.close()
    profiles = guide.groupby(level=args.gene_column,sort=False).median().astype('float32')
    profiles.index = profiles.index.astype(str)
    return profiles.loc[profiles.index != args.control_key_word].copy()


def export(fig, stem, dpi, comparable=False, formats=('png', 'pdf', 'svg')):
    if comparable:
        from .audit_panel_alignment import require_matplotlib_panel_alignment
        fig.canvas.draw()
        require_matplotlib_panel_alignment(fig, json_out=str(stem)+'.alignment.json',
                                          require_panel_labels=False, strict=True)
    for fmt in dict.fromkeys(formats):
        fig.savefig(str(stem)+'.'+fmt, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def resolve_hit_file(path):
    if path is None:
        LOG.info('No hit file specified: using all non-control genes')
        return None
    if not path.exists():
        LOG.warning('Hit file does not exist: %s; using all non-control genes', path)
        return None
    if not path.is_file():
        raise ValueError(f'Hit path must be a file: {path}')
    return path

