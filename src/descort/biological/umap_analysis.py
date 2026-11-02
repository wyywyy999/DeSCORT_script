#!/usr/bin/env python3
"""Gene/guide aggregation, PCA/UMAP, pathway compactness and paired comparison."""
from pathlib import Path
import textwrap
from scipy.spatial.distance import cdist
import argparse
import hashlib
import json
import logging
import pickle
import re
import sys
import tempfile
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import duckdb
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from threadpoolctl import threadpool_limits

from .utils import (PROJECT, WORKSPACE, TUTORIAL_DIR, input_defaults,
                    save_json, get_files, load_features, inspect_input, q, aggregate,
                    export, resolve_hit_file)
HERE = TUTORIAL_DIR
LOG = logging.getLogger('umap_analysis')
LIBRARIES = {'GO_BP': 'GO_Biological_Process_2025', 'GO_CC': 'GO_Cellular_Component_2025',
             'GO_MF': 'GO_Molecular_Function_2025', 'KEGG': 'KEGG_2021_Human'}
SELECTED = [
 ('ANNOTATED','GLYCOSYLATION'), ('ANNOTATED','PROCESS_UTILIZING_AUTOPHAGIC_MECHANISM'),
 ('GO_BP','U2-type Prespliceosome Assembly (GO:1903241)'),
 ('GO_BP','Protein Import Into Mitochondrial Matrix (GO:0030150)'),
 ('GO_BP','Regulation of Angiogenesis (GO:0045765)'),
 ('GO_BP','Sister Chromatid Segregation (GO:0000819)'),
 ('GO_BP','Regulation of Rho Protein Signal Transduction (GO:0035023)'),
 ('GO_BP','Positive Regulation of Biosynthetic Process (GO:0009891)'),
 ('GO_BP','Mitotic Nuclear Division (GO:0140014)'),
 ('GO_CC','Cytosolic Large Ribosomal Subunit (GO:0022625)'),
 ('GO_CC','U6 snRNP (GO:0005688)'), ('GO_MF','snoRNA Binding (GO:0030515)')]


def parser():
    defaults = input_defaults()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--model-input', type=Path, default=defaults.model_input)
    p.add_argument('--original-input', type=Path, default=defaults.original_input)
    p.add_argument('--output-dir', type=Path, default=HERE/'result/01_umap_analysis')
    p.add_argument('--hit-file', type=lambda value: Path(value) if value.strip() else None,
                   nargs='?', const=None, default=None,
                   help='Optional Excel/CSV hit list. Omitted, empty or nonexistent file: use all non-control genes')
    p.add_argument('--hit-column', default=defaults.hit_column)
    p.add_argument('--annotated-gene-sets', type=Path, default=WORKSPACE/'LLLM/annotated_gene_sets.json')
    p.add_argument('--gene-sets-json', type=Path, help='Offline full database->pathway->genes JSON; otherwise fetch/cache the four Enrichr libraries')
    p.add_argument('--library-cache', type=Path, help='Library cache directory; defaults to OUTPUT_DIR/library_cache')
    p.add_argument('--original-features-pkl', type=Path, default=defaults.original_features_pkl)
    p.add_argument('--model-feature-prefix', default=defaults.model_feature_prefix)
    p.add_argument('--model-feature-dim', type=int, default=defaults.model_feature_dim)
    p.add_argument('--gene-column', default=defaults.gene_column)
    p.add_argument('--guide-column', default=defaults.guide_column)
    p.add_argument('--control-key-word', default=defaults.control_key_word)
    p.add_argument('--plates', nargs='*', default=defaults.plates, help='Filename substrings; pass --plates with no values to use all parquet files')
    p.add_argument('--dataset-name', default='A549')
    p.add_argument('--stage', choices=['all','model','original','compare','check','combined-plot','query'], default='all')
    p.add_argument('--model-gene-profiles', type=Path, help='Optional cached gene-level parquet from the exact model inputs; skips aggregation')
    p.add_argument('--original-gene-profiles', type=Path, help='Optional cached gene-level parquet from the exact original inputs; skips aggregation')
    p.add_argument('--model-results', type=Path, help='Existing results directory for compare stage')
    p.add_argument('--original-results', type=Path, help='Existing results directory for compare stage')
    p.add_argument('--corum-file', type=Path, default=WORKSPACE/'LLLM/common_files/CORUM_humanComplexes.txt')
    p.add_argument('--string-file', type=Path, default=WORKSPACE/'LLLM/common_files/STRING_data.csv')
    p.add_argument('--string-threshold', type=int, default=400)
    p.add_argument('--query-gene', nargs='+', default=None,
                   help='Optional query genes; omitted: skip nearest-neighbor analysis and figure 7')
    p.add_argument('--query-top-n', type=int, default=20)
    p.add_argument('--pca-variance', type=float, default=.95)
    p.add_argument('--umap-n-neighbors', type=int, default=10)
    p.add_argument('--umap-min-dist', type=float, default=.05)
    p.add_argument('--umap-metric', default='cosine')
    p.add_argument('--distance-metric', default='cosine')
    p.add_argument('--min-go-kegg-genes', type=int, default=5)
    p.add_argument('--max-go-kegg-genes', type=int, default=200)
    p.add_argument('--min-annotated-genes', type=int, default=2)
    p.add_argument('--max-annotated-genes', type=int)
    p.add_argument('--compact-percentile', type=float, default=100)
    p.add_argument('--top-fraction', type=float, default=.30)
    p.add_argument('--top-rank-by', choices=['model_distance','improvement','original_distance'], default='model_distance')
    p.add_argument('--exclude-databases', nargs='*', default=['ANNOTATED'])
    p.add_argument('--selected-pathways-json', type=Path, help='JSON list of [database, pathway]; default is the notebook list of 12')
    p.add_argument('--skip-pathway-plots', action='store_true', help='Calculate/export all tables, omit individual pathway PNGs')
    p.add_argument('--max-pathway-plots', type=int, default=0, help='Rendering-only cap per non-ANNOTATED database; 0 means all. ANNOTATED always renders all. Tables are never capped.')
    p.add_argument('--pathway-plot-fraction', type=float, default=1.0,
                   help='Rendering-only fraction per non-ANNOTATED database, in (0,1]; select smallest distances first, round up, then apply max-pathway-plots. ANNOTATED always renders all')
    p.add_argument('--show-gene-labels', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--max-gene-labels', type=int, default=40)
    p.add_argument('--png-dpi', type=int, default=300)
    p.add_argument('--comparison-dpi', type=int, default=600)
    p.add_argument('--figure-formats', nargs='+', type=str.lower,
                   choices=['pdf', 'png', 'svg'], default=['png', 'pdf', 'svg'],
                   help='Figure export formats, for example: --figure-formats png or --figure-formats pdf svg')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=defaults.threads)
    p.add_argument('--memory-limit', default=defaults.memory_limit)
    return p










def gene_sets(args):
    if args.gene_sets_json:
        data = json.loads(args.gene_sets_json.read_text())
    else:
        import gseapy as gp
        cache = args.library_cache or args.output_dir/'library_cache'
        cache.mkdir(parents=True, exist_ok=True)
        data = {}
        for db, name in LIBRARIES.items():
            path = cache/f'{name}.json'
            if path.exists():
                data[db] = json.loads(path.read_text())
            else:
                LOG.info('Fetching %s (use --gene-sets-json for offline execution)', name)
                try:
                    data[db] = gp.get_library(name=name, organism='Human')
                except Exception as exc:
                    raise RuntimeError(f'Cannot fetch {name}; provide --gene-sets-json or populate {cache}') from exc
                save_json(path, data[db])
    annotated = json.loads(args.annotated_gene_sets.read_text())
    data['ANNOTATED'] = annotated
    clean = {}
    for db, pathways in data.items():
        clean[db] = {}
        for pathway, value in pathways.items():
            if isinstance(value, dict):
                value = next((value[k] for k in ['genes','Genes','gene','Gene'] if k in value), None)
            if not isinstance(value, list):
                LOG.warning('Skipping invalid gene set: %s / %s',db,pathway)
                continue
            clean[db][pathway] = list(dict.fromkeys(str(v).strip() for v in value if v is not None and str(v).strip()))
    save_json(args.output_dir/'gene_sets_used.json',clean)
    return clean










def select_gene_profiles(profiles, args):
    path = resolve_hit_file(args.hit_file)
    profiles = profiles.loc[profiles.index != args.control_key_word]
    if path is None:
        return profiles.copy(), [], 'all_genes'
    hits = pd.read_excel(path) if path.suffix.lower() in ['.xlsx', '.xls'] else pd.read_csv(path)
    if args.hit_column not in hits.columns:
        raise ValueError(f'{path}: missing hit column {args.hit_column!r}')
    wanted = list(dict.fromkeys(hits[args.hit_column].dropna().astype(str)))
    missing = [g for g in wanted if g not in profiles.index]
    return profiles.loc[[g for g in wanted if g in profiles.index]].copy(), missing, 'hit_list'


def select_pathway_plots(frame, fraction=1.0, maximum=0):
    """Select a deterministic rendering subset without changing analysis tables."""
    if not 0 < fraction <= 1 or maximum < 0:
        raise ValueError('Plot fraction must be in (0,1] and maximum must be nonnegative')
    ranked = frame.sort_values(['mean_distance', 'pathway'], kind='stable')
    if not ranked.empty and ranked['database'].eq('ANNOTATED').all():
        return ranked
    count = int(np.ceil(len(ranked) * fraction))
    if maximum:
        count = min(count, maximum)
    return ranked.head(count)


def analyze(representation, args, sets):
    import umap.umap_ as umap
    out = args.output_dir/representation
    result, figures = out/'results', out/'figures'
    result.mkdir(parents=True,exist_ok=True); figures.mkdir(parents=True,exist_ok=True)
    files = get_files(getattr(args,representation+'_input'),args.plates)
    features, info = inspect_input(files,representation,args,result)
    cached = getattr(args,representation+'_gene_profiles')
    profiles = pd.read_parquet(cached) if cached else aggregate(files,features,args)
    if cached:
        if not profiles.index.is_unique or not set(features) <= set(profiles.columns):
            raise ValueError('Cached gene profiles have invalid index or missing input features')
        profiles = profiles[features].copy()
        profiles.index = profiles.index.astype(str)
        profiles = profiles.loc[profiles.index != args.control_key_word]
        info['cached_gene_profiles'] = str(cached)
        info['aggregation_skipped'] = True
    profiles.to_parquet(result/'all_gene_profiles.parquet')
    hit, missing, selection_mode = select_gene_profiles(profiles, args)
    selected_before_qc = len(hit)
    pd.DataFrame({'gene':missing}).to_csv(result/'missing_hit_genes.csv',index=False)
    invalid = ~np.isfinite(hit.to_numpy()).all(axis=1)
    invalid_genes = hit.index[invalid].tolist()
    hit = hit.loc[~invalid]
    constant = hit.columns[hit.std(axis=0)<=1e-12].tolist()
    hit = hit.drop(columns=constant)
    if len(hit)<3 or hit.shape[1]<2:
        raise ValueError('At least 3 valid selected genes and 2 variable features required')
    hit.to_parquet(result/'hit_gene_profiles.parquet')
    pca = PCA(n_components=args.pca_variance,svd_solver='full')
    with threadpool_limits(limits=args.threads):
        scores = pca.fit_transform(hit.to_numpy(dtype=np.float32)).astype('float32')
    pca_df = pd.DataFrame(scores,index=hit.index,columns=[f'PC{i+1}' for i in range(scores.shape[1])])
    pca_df.to_csv(result/'hit_gene_PCA.csv')
    reducer = umap.UMAP(n_neighbors=args.umap_n_neighbors,min_dist=args.umap_min_dist,
                        n_components=2,metric=args.umap_metric,random_state=args.seed)
    xy = reducer.fit_transform(scores)
    coord = pd.DataFrame(xy,index=hit.index,columns=['UMAP1','UMAP2'])
    coord.index.name='gene';coord.to_csv(result/'global_hit_gene_UMAP.csv')
    fig, ax = plt.subplots(figsize=(7,6.5))
    ax.scatter(xy[:,0],xy[:,1],s=20,color='#777777',alpha=.5,edgecolors='none')
    gene_label = 'hit genes' if selection_mode == 'hit_list' else 'all valid genes'
    ax.set(xlabel='UMAP1',ylabel='UMAP2',title=f'{args.dataset_name} {gene_label}: {representation}')
    fig.tight_layout();export(fig,figures/'00_global_hit_gene_UMAP',args.png_dpi,formats=args.figure_formats)
    rows=[]
    for db,pathways in sets.items():
        for name,genes in pathways.items():
            valid=[g for g in genes if g in pca_df.index]; n=len(valid)
            lower=args.min_annotated_genes if db=='ANNOTATED' else args.min_go_kegg_genes
            upper=args.max_annotated_genes if db=='ANNOTATED' else args.max_go_kegg_genes
            if n<lower or (upper is not None and n>upper): continue
            distance=pairwise_distances(pca_df.loc[valid],metric=args.distance_metric)
            values=distance[np.triu_indices(n,k=1)];values=values[np.isfinite(values)]
            if not len(values):continue
            rows.append(dict(representation=representation,database=db,pathway=name,n_genes=n,n_pairs=len(values),
                             mean_distance=float(values.mean()),median_distance=float(np.median(values)),
                             std_distance=float(values.std()),min_distance=float(values.min()),
                             q25_distance=float(np.quantile(values,.25)),q75_distance=float(np.quantile(values,.75)),
                             max_distance=float(values.max()),genes=';'.join(valid)))
    if not rows:raise RuntimeError('No valid pathways; check hit genes and gene sets')
    table=pd.DataFrame(rows).sort_values(['database','mean_distance']).reset_index(drop=True)
    thresholds={db:float(np.percentile(sub.mean_distance,args.compact_percentile)) for db,sub in table.groupby('database') if db!='ANNOTATED'}
    summary = table.groupby('database').agg(valid_pathways=('pathway','size'), median_n_genes=('n_genes','median'), median_distance=('mean_distance','median')).reset_index()
    summary.to_csv(result/'valid_pathway_summary.csv',index=False)
    dbs = [db for db in LIBRARIES if db in thresholds]
    if dbs:
        fig, axes = plt.subplots(1,len(dbs),figsize=(4*len(dbs),4),squeeze=False)
        for ax,db in zip(axes.ravel(),dbs):
            sub=table[table.database==db]
            ax.hist(sub.mean_distance,bins=35,color='#91b894',edgecolor='white',linewidth=.5)
            ax.axvline(thresholds[db],color='#b45243',linestyle='--',linewidth=1.5)
            ax.set(title=db,xlabel=f'Mean within-pathway {args.distance_metric} distance')
        axes[0,0].set_ylabel('Pathway count')
        fig.subplots_adjust(left=.06,right=.99,bottom=.20,top=.85,wspace=.30)
        export(fig,figures/'01_GO_KEGG_distance_distributions',args.png_dpi,comparable=True,formats=args.figure_formats)
    table['threshold']=table.database.map(thresholds)
    table.to_csv(result/f'pathway_distance_df_{representation}.csv',index=False)
    chosen=table[(table.database=='ANNOTATED')|(table.mean_distance<=table.threshold)].copy()
    chosen['distance_rank']=chosen.groupby('database').mean_distance.rank(method='first').astype(int)
    chosen=chosen.sort_values(['database','distance_rank'])
    chosen.to_csv(result/f'selected_pathway_df_{representation}.csv',index=False)
    pd.DataFrame([dict(database=k,threshold=v,percentile=args.compact_percentile) for k,v in thresholds.items()]).to_csv(result/'GO_KEGG_distance_thresholds.csv',index=False)
    plotted_sets={}
    coordinate_rows=[]
    for _,row in chosen.iterrows():
        genes=row.genes.split(';')
        plotted_sets.setdefault(row.database,{})[row.pathway]=genes
        for gene in genes:
            coordinate_rows.append(dict(database=row.database,pathway=row.pathway,gene=gene,
                                        UMAP1=float(coord.loc[gene,'UMAP1']),UMAP2=float(coord.loc[gene,'UMAP2']),
                                        n_genes=row.n_genes,mean_distance=row.mean_distance,distance_rank=row.distance_rank))
    save_json(result/'plotted_pathway_gene_dict.json',plotted_sets)
    pd.DataFrame(coordinate_rows).to_csv(result/'pathway_gene_UMAP_coordinates.csv',index=False)
    plot_count=0
    rendered_rows=[]
    if not args.skip_pathway_plots:
        xpad=max(np.ptp(xy[:,0])*.05,.01);ypad=max(np.ptp(xy[:,1])*.05,.01)
        for db,sub in chosen.groupby('database'):
            folder=figures/db;folder.mkdir(exist_ok=True)
            sub=select_pathway_plots(sub,args.pathway_plot_fraction,args.max_pathway_plots)
            LOG.info('%s %s: rendering %d / %d selected pathways',representation,db,len(sub),int((chosen.database==db).sum()))
            rendered_rows.append(sub)
            for _,row in sub.iterrows():
                genes=row.genes.split(';');v=coord.loc[genes]
                fig,ax=plt.subplots(figsize=(7.2,6.5))
                ax.scatter(xy[:,0],xy[:,1],s=18,color='#d7d7d7',alpha=.4,edgecolors='none')
                ax.scatter(v.UMAP1,v.UMAP2,s=72,color='#4f8465',edgecolors='white',linewidths=.65)
                ax.scatter(v.UMAP1.median(),v.UMAP2.median(),s=125,marker='X',color='#b34e3d',edgecolors='white')
                if args.show_gene_labels and len(genes)<=args.max_gene_labels:
                    for gene in genes:ax.annotate(gene,coord.loc[gene],xytext=(3,3),textcoords='offset points',fontsize=6.8)
                title=re.sub(r'\s*\(GO:\d+\)','',row.pathway).replace('_',' ')
                import textwrap
                ax.set(title='\n'.join(textwrap.wrap(title,55)),xlabel='UMAP1',ylabel='UMAP2',
                       xlim=(xy[:,0].min()-xpad,xy[:,0].max()+xpad),ylim=(xy[:,1].min()-ypad,xy[:,1].max()+ypad))
                ax.text(.985,.985,f'{db}\nGenes = {row.n_genes}\nMean distance = {row.mean_distance:.3f}\nMedian distance = {row.median_distance:.3f}',transform=ax.transAxes,ha='right',va='top',fontsize=8.5,bbox=dict(facecolor='white',alpha=.94,edgecolor='#bdbdbd'))
                filename=re.sub(r'[^\w.-]+','_',row.pathway)[:140]
                fig.tight_layout();export(fig,folder/f'{row.distance_rank:03d}_{filename}',args.png_dpi,formats=args.figure_formats);plot_count+=1
    rendered = pd.concat(rendered_rows, ignore_index=True) if rendered_rows else chosen.iloc[:0].copy()
    rendered.to_csv(result/'rendered_pathways.csv',index=False)
    plot_combined_pathways(coord, plotted_sets, figures, args, representation)
    save_json(result/'run_manifest.json',dict(input=info,valid_hit_genes=len(hit),missing_hit_genes=missing,
              gene_selection_mode=selection_mode,selected_genes_before_qc=selected_before_qc,
              valid_selected_genes=len(hit),
              invalid_hit_genes_removed=invalid_genes,constant_features_removed=constant,pca_components=scores.shape[1],
              explained_variance=float(pca.explained_variance_ratio_.sum()),all_pathways=len(table),selected_pathways=len(chosen),
              individual_pathways_rendered=plot_count,parameters=vars(args)))
    LOG.info('%s: %d selected genes (%s), %d PCA components, %d pathways',representation,len(hit),selection_mode,scores.shape[1],len(table))
    return result


def paired_summary(frame):
    delta=frame.mean_distance_original-frame.mean_distance_model
    if not len(frame):return dict(n_pathways=0,wilcoxon_p=None)
    p=1.0 if np.all(delta.to_numpy()==0) else float(wilcoxon(frame.mean_distance_original,frame.mean_distance_model,alternative='greater').pvalue)
    return dict(n_pathways=len(frame),mean_model=float(frame.mean_distance_model.mean()),mean_original=float(frame.mean_distance_original.mean()),
                median_model=float(frame.mean_distance_model.median()),median_original=float(frame.mean_distance_original.median()),
                model_better_fraction=float((delta>0).mean()),wilcoxon_p=p)


def add_differences(frame):
    frame=frame.copy()
    frame['delta_distance']=frame.mean_distance_original-frame.mean_distance_model
    frame['relative_improvement_percent']=np.where(frame.mean_distance_original!=0,frame.delta_distance/frame.mean_distance_original*100,np.nan)
    frame['model_better']=frame.delta_distance>0
    return frame


def paired_plot(frame, stem, args, grouped=False):
    if frame.empty:return
    dbs=[v for v in ['GO_BP','GO_CC','KEGG','GO_MF'] if v in set(frame.database)] if grouped else [None]
    if not dbs:return
    fig,axes=plt.subplots(1,len(dbs),figsize=(3.3*len(dbs),5.3) if grouped else (5.2,5.4),squeeze=False)
    # Random jitter changes only point positions, never observed distance values.
    # No simulated data, seed/fold aggregate, or uncertainty interval is plotted.
    rng=np.random.default_rng(args.seed)
    for ax,db in zip(axes.ravel(),dbs):
        sub=frame if db is None else frame[frame.database==db]
        model=sub.mean_distance_model.to_numpy();original=sub.mean_distance_original.to_numpy()
        box=ax.boxplot([model,original],positions=[0,.6],widths=.28,patch_artist=True,showfliers=False,medianprops=dict(color='#20262a'))
        for i,(vals,color) in enumerate([(model,'#BC3C28'),(original,'#0072B5')]):
            box['boxes'][i].set_facecolor(color);box['boxes'][i].set_alpha(.82)
            ax.scatter(rng.normal(i*.6,.035,len(vals)),vals,s=18,color=color,alpha=.3,edgecolors='none')
            ax.scatter(i*.6,vals.mean(),marker='D',s=34,color=color,edgecolors='black',linewidths=.5,zorder=5)
        lo=min(model.min(),original.min());hi=max(model.max(),original.max());span=max(hi-lo,.05)
        stats=paired_summary(sub);p=stats['wilcoxon_p']
        star='***' if p<.001 else '**' if p<.01 else '*' if p<.05 else 'n.s.'
        y=hi+.16*span
        ax.plot([0,0,.6,.6],[y-.02*span,y,y,y-.02*span],color='#586168',lw=.9)
        ax.text(.3,y+.08*span,star,ha='center',fontsize=12)
        for i,vals in enumerate([model,original]):ax.text(i*.6,hi+.025*span,f'{vals.mean():.3f}',ha='center',fontsize=9)
        ax.set(xlim=(-.5,1.1),ylim=(lo-.12*span,hi+.45*span),xticks=[0,.6],xticklabels=['DeSCORT','Original'],title=f'{db or "Selected pathways"} (n = {len(sub)})')
        if not grouped:ax.set_ylabel('Mean within-pathway cosine distance')
    if grouped:fig.supylabel('Mean within-pathway cosine distance',fontsize=12)
    fig.subplots_adjust(left=.08 if grouped else .18,right=.98,bottom=.16,top=.83,wspace=.30)
    export(fig,stem,args.comparison_dpi,comparable=grouped,formats=args.figure_formats)


def compare(model_dir, original_dir, args):
    out=args.output_dir/'comparison';out.mkdir(parents=True,exist_ok=True)
    def load(folder,rep,selected=False):
        prefix='selected_pathway_df' if selected else 'pathway_distance_df'
        d=pd.read_csv(folder/f'{prefix}_{rep}.csv')
        if d.duplicated(['database','pathway']).any():raise ValueError('Duplicate pathway keys')
        return d
    def merge(original,model):
        cols=['database','pathway','n_genes','mean_distance','median_distance','genes']
        d=original[cols].merge(model[cols],on=['database','pathway'],validate='one_to_one',suffixes=('_original','_model'))
        def genes(v):return frozenset(g.strip() for g in str(v).split(';') if g.strip())
        d['same_genes']=[genes(a)==genes(b) for a,b in zip(d.genes_original,d.genes_model)]
        return add_differences(d.dropna(subset=['mean_distance_original','mean_distance_model']))
    orig,model=load(original_dir,'original'),load(model_dir,'model')
    selected=json.loads(args.selected_pathways_json.read_text()) if args.selected_pathways_json else SELECTED
    allpairs=merge(orig,model)
    manual=pd.DataFrame(selected,columns=['database','pathway']).merge(allpairs,on=['database','pathway'],how='left',validate='one_to_one')
    manual.to_csv(out/'manual_selected_pathways.csv',index=False)
    # Preserve the notebook's warning-only behavior for the manual list.
    manual_valid=manual.dropna(subset=['mean_distance_original','mean_distance_model'])
    paired_plot(manual_valid,out/'manual_selected_boxplot',args)
    common=merge(load(original_dir,'original',True),load(model_dir,'model',True))
    common.to_csv(out/'common_pathways_gene_set_check.csv',index=False)
    full=common[common.same_genes & ~common.database.isin(args.exclude_databases)].sort_values('delta_distance',ascending=False).reset_index(drop=True)
    if full.empty:raise ValueError('No strict pathways after matching/exclusion')
    full.to_csv(out/'all_strict_pathways.csv',index=False)
    col,asc={'model_distance':('mean_distance_model',True),'improvement':('delta_distance',False),'original_distance':('mean_distance_original',True)}[args.top_rank_by]
    n=min(len(full),max(2,int(np.ceil(len(full)*args.top_fraction))))
    top=full.sort_values(col,ascending=asc).head(n).reset_index(drop=True)
    top.to_csv(out/'top_strict_pathways.csv',index=False)
    paired_plot(top,out/'selected_pathway_model_vs_original_boxplot',args)
    paired_plot(top,out/'selected_pathway_model_vs_original_by_database',args,grouped=True)
    pd.DataFrame([dict(database=db,**paired_summary(sub)) for db,sub in top.groupby('database')]).to_csv(out/'by_database_summary.csv',index=False)
    summary=dict(common_selected_pathways=len(common),different_gene_sets=int((~common.same_genes).sum()),excluded_databases=args.exclude_databases,
                 all_strict=paired_summary(full),top_subset=paired_summary(top),manual_subset=paired_summary(manual_valid),
                 top_fraction=args.top_fraction,top_rank_by=args.top_rank_by,
                 interpretation='Top subset is selected using observed distances; its P values are exploratory. Overlapping gene sets are not independent. Compactness alone does not establish biological specificity.',
                 model_results=str(model_dir),original_results=str(original_dir))
    save_json(out/'comparison_summary.json',summary);LOG.info('Comparison: %s',summary)


def majority_cluster_anchor(xy, fraction=0.60):
    if len(xy) == 1:
        return xy[0]
    distances = cdist(xy, xy)
    neighbour_count = min(len(xy), max(3, int(np.ceil(fraction * len(xy)))))
    nearest = np.argsort(distances, axis=1)[:, :neighbour_count]
    compactness = np.take_along_axis(distances, nearest, axis=1).sum(axis=1)
    medoid = int(np.argmin(compactness))
    return np.median(xy[nearest[medoid]], axis=0)

def pathway_label(database, pathway_name, width=22):
    name = re.sub(r'\s*\(GO:\d+\)', '', str(pathway_name)).replace('_', ' ')
    return "\n".join(textwrap.wrap(name, width=width, break_long_words=False, break_on_hyphens=False))

def plot_combined_pathways(coord, plotted_sets, figures, args, representation):
    """Overlay the notebook's selected pathways on unchanged global UMAP coordinates.

    Missing/filtered pathways are logged and recorded; coordinates are never refit.
    Overlapping genes retain the notebook's successive colored overlays.
    """
    requested = json.loads(args.selected_pathways_json.read_text()) if args.selected_pathways_json else SELECTED
    available, audit = [], []
    for database, name in requested:
        genes = [g for g in plotted_sets.get(database, {}).get(name, []) if g in coord.index]
        audit.append(dict(database=database, pathway=name, n_genes=len(genes), plotted=bool(genes)))
        if genes:
            available.append((database, name))
        else:
            LOG.warning('%s: combined UMAP omits unavailable pathway %s / %s', representation, database, name)
    figures.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit, columns=['database','pathway','n_genes','plotted']).to_csv(
        figures/'selected_plotted_pathways_combined_UMAP_selection.csv', index=False)
    if not available:
        LOG.warning('%s: no selected pathways available for combined UMAP', representation)
        return pd.DataFrame()
    SELECTED_PATHWAYS = available
    plotted_pathway_gene_dict = plotted_sets
    umap_df = coord
    MULTI_PATHWAY_COLORS = ['#3C7D74','#C17A52','#6678A5','#9A6C82','#76955D','#B45A5A',
                           '#8C8068','#5E8C9A','#A58A4A','#7C6D9A','#4F8A6D','#B06F7A']
    # Keep a pathway's color stable when a preceding pathway is unavailable.
    palette = MULTI_PATHWAY_COLORS
    MULTI_PATHWAY_COLORS = [palette[i % len(palette)] if i < len(palette) else
                           matplotlib.colors.to_hex(plt.get_cmap('tab20')((i-len(palette)) % 20))
                           for i, pair in enumerate(requested) if tuple(pair) in available]
    POINT_SIZE = 12
    selected = []
    for (database, pathway_name) in SELECTED_PATHWAYS:
        genes = plotted_pathway_gene_dict[database][pathway_name]
        genes = [gene for gene in genes if gene in umap_df.index]
        if genes:
            selected.append((database, pathway_name, genes))
    (fig, ax) = plt.subplots(figsize=(7.2, 5.5), dpi=args.png_dpi, facecolor='white')
    fig.subplots_adjust(left=0.095, right=0.965, top=0.9, bottom=0.135)
    ax.scatter(umap_df['UMAP1'], umap_df['UMAP2'], s=POINT_SIZE, color='#CBD0D4', alpha=0.4, edgecolors='none', zorder=1)
    anchors = []
    colored_gene_set = set()
    for (index, (database, pathway_name, genes)) in enumerate(selected):
        color = MULTI_PATHWAY_COLORS[index]
        xy = umap_df.loc[genes, ['UMAP1', 'UMAP2']].to_numpy()
        ax.scatter(xy[:, 0], xy[:, 1], s=POINT_SIZE, color=color, alpha=0.95, edgecolors='none', zorder=3)
        colored_gene_set.update(genes)
        anchors.append(majority_cluster_anchor(xy, fraction=0.6))
    (xmin, xmax) = (umap_df['UMAP1'].min(), umap_df['UMAP1'].max())
    (ymin, ymax) = (umap_df['UMAP2'].min(), umap_df['UMAP2'].max())
    ax.set_xlim(xmin - 0.42, xmax + 0.42)
    ax.set_ylim(ymin - 0.38, ymax + 0.38)
    ax.set_xlabel('UMAP 1', fontsize=9)
    ax.set_ylabel('UMAP 2', fontsize=9)
    ax.tick_params(labelsize=8, width=0.7, length=3)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_linewidth(0.7)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    colored_genes = list(colored_gene_set)
    colored_xy = umap_df.loc[colored_genes, ['UMAP1', 'UMAP2']].to_numpy()
    colored_pixels = ax.transData.transform(colored_xy)
    pixels_per_point = fig.dpi / 72.0
    directions = [(0, 1), (1, 1), (-1, 1), (1, 0), (-1, 0), (1, -1), (-1, -1), (0, -1)]
    candidate_offsets = [(distance * dx, distance * dy) for distance in (10, 20, 32, 46, 61, 78) for (dx, dy) in directions]
    accepted_bboxes = []
    label_rows = []
    anchor_pixels = ax.transData.transform(np.asarray(anchors))
    if len(anchors) > 1:
        crowding = cdist(anchor_pixels, anchor_pixels)
        np.fill_diagonal(crowding, np.inf)
        placement_order = np.argsort(np.min(crowding, axis=1))
    else:
        placement_order = np.arange(len(anchors))
    for index in placement_order:
        (database, pathway_name, genes) = selected[index]
        color = MULTI_PATHWAY_COLORS[index]
        rgb = np.asarray(matplotlib.colors.to_rgb(color))
        text_color = tuple(np.clip(rgb * 0.76, 0, 1))
        label = ax.text(anchors[index][0], anchors[index][1], pathway_label(database, pathway_name, width=22), ha='center', va='center', fontsize=7.4, fontweight='semibold', linespacing=1.08, color=text_color, zorder=5)
        best = None
        for (dx_pt, dy_pt) in candidate_offsets:
            candidate_pixel = anchor_pixels[index] + pixels_per_point * np.array([dx_pt, dy_pt])
            candidate_data = ax.transData.inverted().transform(candidate_pixel)
            label.set_position(candidate_data)
            bbox = label.get_window_extent(renderer).expanded(1.08, 1.14)
            inside = bbox.x0 >= ax.bbox.x0 + 3 and bbox.x1 <= ax.bbox.x1 - 3 and (bbox.y0 >= ax.bbox.y0 + 3) and (bbox.y1 <= ax.bbox.y1 - 3)
            if not inside:
                continue
            overlap = 0.0
            for prior in accepted_bboxes:
                overlap_x = max(0, min(bbox.x1, prior.x1) - max(bbox.x0, prior.x0))
                overlap_y = max(0, min(bbox.y1, prior.y1) - max(bbox.y0, prior.y0))
                overlap += overlap_x * overlap_y
            point_cover = np.count_nonzero((colored_pixels[:, 0] >= bbox.x0) & (colored_pixels[:, 0] <= bbox.x1) & (colored_pixels[:, 1] >= bbox.y0) & (colored_pixels[:, 1] <= bbox.y1))
            distance_pt = np.hypot(dx_pt, dy_pt)
            cost = 35 * overlap + 20 * point_cover + 0.8 * distance_pt
            if best is None or cost < best[0]:
                best = (cost, candidate_data, bbox, overlap, point_cover, distance_pt)
        if best is None:
            best_position = anchors[index]
            label.set_position(best_position)
            bbox = label.get_window_extent(renderer).expanded(1.08, 1.14)
            best = (np.inf, best_position, bbox, np.nan, np.nan, 0)
        label.set_position(best[1])
        accepted_bboxes.append(best[2])
        label_rows.append({'database': database, 'pathway': pathway_name, 'n_genes': len(genes), 'anchor_umap1': anchors[index][0], 'anchor_umap2': anchors[index][1], 'label_umap1': best[1][0], 'label_umap2': best[1][1], 'label_shift_pt': best[5], 'label_overlap_px2': best[3], 'colored_points_covered': best[4]})
    direct_label_table = pd.DataFrame(label_rows).sort_values(['database', 'pathway']).reset_index(drop=True)
    stem = figures / 'selected_plotted_pathways_combined_UMAP_direct_labels'
    direct_label_table.to_csv(stem.with_suffix('.labels.csv'), index=False)
    export(fig, stem, args.png_dpi, formats=args.figure_formats)
    LOG.info('Saved combined pathway UMAP: %s (%d pathways)', stem, len(selected))
    return direct_label_table


def redraw_combined_pathways(result, args, representation):
    """Read exported coordinates and selected gene sets without rerunning PCA/UMAP."""
    coord = pd.read_csv(result/'global_hit_gene_UMAP.csv', index_col=0)
    coord.index = coord.index.astype(str)
    sets = json.loads((result/'plotted_pathway_gene_dict.json').read_text())
    return plot_combined_pathways(coord, sets, result.parent/'figures', args, representation)


# Figure 7 computes neighbors directly from this workflow's profiles and annotations.
def select_neighbor_query(table, gene, top_n):
    if top_n < 1:
        raise ValueError('query-top-n must be positive')
    gene = str(gene).strip().upper()
    query = table.loc[table.query_gene.astype(str).str.upper() == gene].nsmallest(top_n,'rank')
    if query.empty:
        raise KeyError(f'Gene not found in neighbor table: {gene}')
    return query.reset_index(drop=True)


def save_neighbor_figure(fig, stem, qa, args):
    from .audit_panel_alignment import require_matplotlib_panel_alignment
    require_matplotlib_panel_alignment(fig, json_out=str(qa/(stem.name+'.alignment.json')),
        overlay_svg=str(qa/(stem.name+'.alignment.svg')), require_panel_labels=False, strict=True)
    export(fig,stem,args.png_dpi,formats=args.figure_formats)


def plot_query_example(neighbor_table, gene, args, figures, qa):
    from matplotlib.lines import Line2D
    query = select_neighbor_query(neighbor_table, gene, args.query_top_n).sort_values('rank', ascending=False)
    colors = np.where(query['biological_support_count'] > 0, '#BC3C28', '#AEB5BA')
    fig, ax = plt.subplots(figsize=(4.6, 5.2))
    fig.subplots_adjust(left=0.30, right=0.98, bottom=0.19, top=0.90)
    ax.barh(query['neighbor_gene'], query['model_cosine'], color=colors)
    ax.set(xlabel='DeSCORT cosine similarity', ylabel='Neighbour gene',
           title=f'Nearest neighbours of {gene}')
    handles = [
        Line2D([0], [0], marker='s', color='none', markerfacecolor='#BC3C28',
               markeredgecolor='none', label='Known biological support'),
        Line2D([0], [0], marker='s', color='none', markerfacecolor='#AEB5BA',
               markeredgecolor='none', label='No support in tested resources'),
    ]
    ax.legend(
        handles=handles, frameon=False, loc='upper center',
        bbox_to_anchor=(0.5, -0.13), ncol=1,
    )
    save_neighbor_figure(fig, figures/f'figure7_query_example_{gene}', qa, args)



def neighbor_profiles(representation, args, analysis_results=None):
    """Prepare profiles from this 01 run or its explicit raw/profile input."""
    if analysis_results is not None:
        path = analysis_results/'hit_gene_profiles.parquet'
        frame = pd.read_parquet(path)
        return frame, dict(source=str(path), prepared_by_current_analysis=True)
    cached = getattr(args,representation+'_gene_profiles')
    if cached:
        frame = pd.read_parquet(cached)
        info = dict(source=str(cached), explicit_profile_input=True)
    else:
        files = get_files(getattr(args,representation+'_input'),args.plates)
        features, info = inspect_input(files,representation,args)
        frame = aggregate(files,features,args)
    frame.index=frame.index.astype(str)
    if not frame.index.is_unique:
        raise ValueError(f'{representation}: duplicate gene profile identifiers')
    selected,missing,mode=select_gene_profiles(frame,args)
    info.update(missing_genes=missing,selection_mode=mode)
    return selected,info


def annotate_neighbor_support(table, genes, sets, args):
    """Evaluate CORUM, STRING and GO/KEGG support using 01 annotation inputs."""
    from collections import defaultdict
    universe=set(genes)
    membership={db:defaultdict(set) for db in ['CORUM',*LIBRARIES]}
    corum=pd.read_csv(args.corum_file,sep='\t')
    if 'Organism' in corum:
        corum=corum.loc[corum.Organism.astype(str).str.contains('Human',case=False,na=False)]
    for row in corum.to_dict('records'):
        members={g.strip() for g in str(row.get('subunits(Gene name)','')).split(';')} & universe
        if len(members)<2:continue
        term=str(row.get('ComplexName','CORUM complex'))
        for gene in members:membership['CORUM'][gene].add(term)
    for db in LIBRARIES:
        for term,members in sets.get(db,{}).items():
            members=set(map(str,members)) & universe
            if len(members)<args.min_go_kegg_genes or len(members)>args.max_go_kegg_genes:continue
            for gene in members:membership[db][gene].add(term)
    wanted={tuple(sorted((a,b))) for a,b in zip(table.query_gene,table.neighbor_gene)}
    supported=set()
    for chunk in pd.read_csv(args.string_file,usecols=['protein1','protein2','combined_score'],chunksize=500000):
        chunk=chunk.loc[(chunk.combined_score>=args.string_threshold)
                        & chunk.protein1.isin(universe) & chunk.protein2.isin(universe)]
        supported.update(tuple(sorted((a,b))) for a,b in zip(chunk.protein1,chunk.protein2)
                         if tuple(sorted((a,b))) in wanted)
    rows=[]
    for row in table.to_dict('records'):
        gene,neighbor=row['query_gene'],row['neighbor_gene']
        row['CORUM_supported']=bool(membership['CORUM'][gene] & membership['CORUM'][neighbor])
        row['STRING400_supported']=tuple(sorted((gene,neighbor))) in supported
        for db in LIBRARIES:
            shared=sorted(membership[db][gene] & membership[db][neighbor])
            row[f'shared_{db}_n']=len(shared);row[f'shared_{db}_terms']='; '.join(shared[:5])
        row['biological_support_count']=int(row['CORUM_supported'])+int(row['STRING400_supported'])+sum(row[f'shared_{db}_n']>0 for db in LIBRARIES)
        rows.append(row)
    return pd.DataFrame(rows)


def run_neighbor_plots(args, model_results=None, original_results=None):
    """Compute figure7 independently from 01 profiles, PCA and annotation inputs."""
    if not args.query_gene:
        LOG.info('No query genes specified; skipping figure 7')
        return None
    if args.query_top_n<1:raise ValueError('query-top-n must be positive')
    out=args.output_dir/'neighbor_queries'
    figures,results,qa=out/'figures',out/'results',out/'qa'
    for folder in [figures,results,qa]:folder.mkdir(parents=True,exist_ok=True)
    model,model_info=neighbor_profiles('model',args,model_results)
    original,original_info=neighbor_profiles('original',args,original_results)
    common=model.index.intersection(original.index,sort=False)
    model=model.loc[common];original=original.loc[common]
    finite=np.isfinite(model.to_numpy()).all(axis=1)&np.isfinite(original.to_numpy()).all(axis=1)
    excluded=common[~finite].tolist();model=model.loc[finite];original=original.loc[finite]
    model=model.loc[:,model.std(axis=0)>1e-12];original=original.loc[:,original.std(axis=0)>1e-12]
    genes=model.index.to_numpy(dtype=str)
    if len(genes)<3:raise ValueError('At least three common finite genes are required')
    lookup={g.upper():i for i,g in enumerate(genes)}
    if len(lookup)!=len(genes):raise ValueError('Case-insensitive duplicate gene identifiers')
    requested=list(dict.fromkeys(str(g).strip().upper() for g in args.query_gene))
    missing=[g for g in requested if g not in lookup];found=[g for g in requested if g in lookup]
    pd.DataFrame({'gene':missing}).to_csv(results/'query_missing_genes.csv',index=False)
    if not found:raise ValueError('None of the requested genes occur in the common profile inputs')
    for gene in missing:LOG.warning('Query gene missing from 01 inputs: %s',gene)
    spaces={};components={}
    for rep,frame in [('model',model),('original',original)]:
        pca=PCA(n_components=args.pca_variance,svd_solver='full')
        with threadpool_limits(limits=args.threads):z=pca.fit_transform(frame.to_numpy(dtype=np.float32)).astype(np.float32)
        spaces[rep]=z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
        components[rep]=int(pca.n_components_)
        pd.DataFrame(z,index=genes).to_parquet(results/f'{rep}_gene_PCA.parquet')
    count=min(args.query_top_n,len(genes)-1)
    rows=[]
    for gene in found:
        i=lookup[gene]
        similarity=spaces['model'] @ spaces['model'][i]
        similarity[i]=-np.inf
        neighbors=np.argsort(-similarity,kind='stable')[:count]
        for rank,j in enumerate(neighbors,start=1):
            rows.append(dict(query_gene=genes[i],neighbor_gene=genes[j],rank=rank,
                             model_cosine=float(similarity[j]),
                             original_cosine=float(np.dot(spaces['original'][i],spaces['original'][j]))))
    table=annotate_neighbor_support(pd.DataFrame(rows),genes,gene_sets(args),args)
    table.to_parquet(results/'gene_neighbor_rank.parquet',index=False)
    table.to_csv(results/'gene_neighbor_rank.csv',index=False)
    table.to_csv(results/'query_all_genes.csv',index=False)
    with matplotlib.rc_context({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
                                 'font.size':7,'axes.labelsize':8,'axes.titlesize':9,
                                 'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,
                                 'pdf.fonttype':42,'svg.fonttype':'none'}):
        for gene in found:
            select_neighbor_query(table,gene,count).to_csv(results/f'query_{gene}.csv',index=False)
            plot_query_example(table,gene,args,figures,qa)
    save_json(results/'figure7_manifest.json',dict(model_input=model_info,original_input=original_info,
              query_genes=found,missing_genes=missing,requested_top_n=args.query_top_n,effective_top_n=count,
              common_finite_genes=len(genes),excluded_nonfinite_genes=excluded,
              pca_components=components,pca_variance=args.pca_variance,rankings_recomputed=True,
              similarity='Cosine on independently fit, unscaled PCA scores; self excluded',
              support_resources=dict(CORUM=str(args.corum_file),STRING=str(args.string_file),
                                     STRING_threshold=args.string_threshold,GO_KEGG='01 gene_sets_used.json; current size filters'),
              depends_on_03_results=False))
    LOG.info('Figure7 computed independently for %s (%d common genes); output: %s',', '.join(found),len(genes),out)
    return out
