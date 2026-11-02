#!/usr/bin/env python3
"""Compare common hit-gene profiles using PCA and background-corrected CORUM/STRING similarity."""
from pathlib import Path
from itertools import combinations, islice
import argparse
import json
import logging
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from . import utils as io
from .utils import PROJECT, WORKSPACE, TUTORIAL_DIR, RESOURCE_ROOT
HERE = TUTORIAL_DIR
LOG=logging.getLogger('corum_string')
COLORS={'LayerNorm':'#3568B0','Original':'#C94F4F'}
DISPLAY={'LayerNorm':'DeSCORT + LayerNorm','Original':'Original morphology'}
STRING_LABELS=['0.0–0.2','0.2–0.4','0.4–0.6','0.6–0.8','0.8–1.0']


def parser():
    defaults=io.input_defaults()
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for key in ['model_input','original_input','original_features_pkl','model_feature_prefix','model_feature_dim',
                'gene_column','guide_column','control_key_word','hit_column','threads','memory_limit']:
        value=getattr(defaults,key)
        p.add_argument('--'+key.replace('_','-'),type=Path if isinstance(value,Path) else type(value),default=value)
    p.set_defaults(original_features_pkl=RESOURCE_ROOT/'feature_intersection_union/cpg0021_A549/cpg0021_A549_union_feature_1382.pkl')
    p.add_argument('--hit-file',type=lambda value: Path(value) if value.strip() else None,
                   nargs='?', const=None, default=None,
                   help='Optional hit list; omitted, empty or nonexistent: select all common non-control genes before PCA')
    p.add_argument('--corum-file',type=Path,default=WORKSPACE/'LLLM/common_files/CORUM_humanComplexes.txt')
    p.add_argument('--string-file',type=Path,default=WORKSPACE/'LLLM/common_files/STRING_data.csv')
    p.add_argument('--corum-gene-column',default='subunits(Gene name)')
    p.add_argument('--corum-min-coverage',type=float,default=.333,help='Retain complexes with hit/full-member fraction strictly greater than this value')
    p.add_argument('--plates',nargs='*',default=defaults.plates)
    p.add_argument('--model-gene-profiles',type=Path,help='Optional exact-input gene-profile cache; otherwise aggregate raw model parquet')
    p.add_argument('--original-gene-profiles',type=Path,help='Optional exact-input gene-profile cache; otherwise aggregate raw original parquet')
    p.add_argument('--output-dir',type=Path,default=HERE/'result/02_CORUM_STRING')
    p.add_argument('--dataset-name',default='A549')
    p.add_argument('--pca-variance',type=float,default=.95)
    p.add_argument('--seed',type=int,default=20260918)
    p.add_argument('--string-chunk-size',type=int,default=500000)
    p.add_argument('--pair-chunk-size',type=int,default=50000,help='Maximum number of PCA gene pairs scored at once')
    p.add_argument('--string-low-score',type=int,default=300)
    p.add_argument('--string-high-score',type=int,default=700)
    p.add_argument('--figure-formats',nargs='+',type=str.lower,choices=['png','pdf','svg'],default=['png','pdf'])
    p.add_argument('--png-dpi',type=int,default=300)
    p.add_argument('--stage',choices=['all','check','corum-plot','string-plot'],default='all',
                   help='corum-plot/string-plot redraw the requested figure from its saved pair table')
    return p


def collect_profiles(args, result):
    profiles={};input_info={}
    for rep,key in [('LayerNorm','model'),('Original','original')]:
        files=io.get_files(getattr(args,key+'_input'),args.plates)
        folder=result/key;folder.mkdir(exist_ok=True)
        features,info=io.inspect_input(files,key,args,folder)
        cached=getattr(args,key+'_gene_profiles')
        frame=pd.read_parquet(cached) if cached else io.aggregate(files,features,args)
        if not frame.index.is_unique or not set(features)<=set(frame.columns):
            raise ValueError(f'{rep}: invalid gene index or missing feature columns in profiles')
        frame=frame[features].copy();frame.index=frame.index.astype(str)
        frame=frame.loc[frame.index!=args.control_key_word]
        info['gene_profiles_cache']=str(cached) if cached else None
        info['aggregated_genes']=len(frame)
        profiles[rep]=frame;input_info[rep]=info
        frame.to_parquet(folder/'all_gene_profiles.parquet')
    common=profiles['LayerNorm'].index.intersection(profiles['Original'].index,sort=False)
    hit_path=io.resolve_hit_file(args.hit_file)
    if hit_path is None:
        requested=set(common)
        args.gene_selection_mode='all_genes'
    else:
        hit_table=pd.read_excel(hit_path) if hit_path.suffix.lower() in ['.xlsx','.xls'] else pd.read_csv(hit_path)
        if args.hit_column not in hit_table:raise ValueError(f'Missing hit column {args.hit_column}')
        requested=set(hit_table[args.hit_column].dropna().astype(str))
        args.gene_selection_mode='hit_list'
    candidates=[g for g in common if g in requested]
    finite=np.ones(len(candidates),dtype=bool)
    for frame in profiles.values():finite &= np.isfinite(frame.loc[candidates].to_numpy()).all(axis=1)
    genes=[g for g,valid in zip(candidates,finite) if valid]
    if len(genes)<3:raise ValueError('At least three common finite selected genes required')
    pd.DataFrame({'gene':sorted(requested-set(common))}).to_csv(result/'missing_requested_hits.csv',index=False)
    pd.DataFrame({'gene':[g for g,valid in zip(candidates,finite) if not valid]}).to_csv(result/'nonfinite_hits_removed.csv',index=False)
    pd.DataFrame({'gene':genes}).to_csv(result/'common_hit_genes.csv',index=False)
    pd.DataFrame({'gene':genes}).to_csv(result/'common_selected_genes.csv',index=False)
    selected={rep:frame.loc[genes].copy() for rep,frame in profiles.items()}
    for rep,frame in selected.items():frame.to_parquet(result/('model' if rep=='LayerNorm' else 'original')/'hit_gene_profiles.parquet')
    io.save_json(result/'input_qc.json',dict(inputs=input_info,common_genes=len(common),requested_hits=len(requested),
                 common_hits_before_qc=len(candidates),common_finite_hits=len(genes),
                 gene_selection_mode=args.gene_selection_mode,fit_order='select_genes_then_PCA'))
    LOG.info('Common genes %d; common finite selected genes %d (%s)',len(common),len(genes),args.gene_selection_mode)
    return selected,genes


def build_pairs(genes,args,result):
    hits=set(genes);corum=pd.read_csv(args.corum_file,sep='\t')
    if args.corum_gene_column not in corum:raise ValueError('Missing CORUM subunit column')
    positives=set();universe=set();complexes=[]
    for value in corum[args.corum_gene_column].dropna():
        full={g.strip() for g in str(value).split(';') if g.strip()};members=sorted(full & hits)
        universe.update(members)  # Preserve the notebook universe, including sub-threshold complexes.
        if full and members and len(members)/len(full)>args.corum_min_coverage:
            positives.update(combinations(members,2));complexes.append(members)
    all_pairs=list(combinations(sorted(universe),2))
    pos=sorted(positives);neg=[pair for pair in all_pairs if pair not in positives]
    if not pos or not neg:raise ValueError('CORUM needs both positive and non-CORUM background pairs')
    chunks=[]
    for chunk in pd.read_csv(args.string_file,usecols=['protein1','protein2','combined_score'],chunksize=args.string_chunk_size):
        keep=chunk.protein1.isin(hits)&chunk.protein2.isin(hits)&(chunk.protein1<chunk.protein2)
        if keep.any():chunks.append(chunk.loc[keep].copy())
    if not chunks:raise ValueError('No STRING edges between common hit genes')
    edges=pd.concat(chunks,ignore_index=True).drop_duplicates(['protein1','protein2']).reset_index(drop=True)
    if not np.isfinite(edges.combined_score).all() or not edges.combined_score.between(0,1000).all():
        raise ValueError('STRING combined_score must be finite and in [0,1000]')
    edges['confidence_interval']=pd.cut(edges.combined_score,[0,200,400,600,800,1000],labels=STRING_LABELS,include_lowest=True,right=True)
    edges.to_csv(result/'STRING_edges.csv',index=False)
    pd.DataFrame(pos,columns=['gene1','gene2']).to_csv(result/'CORUM_positive_pairs.csv',index=False)
    pd.DataFrame(neg,columns=['gene1','gene2']).to_csv(result/'CORUM_background_pairs.csv',index=False)
    summary=dict(common_hits=len(genes),corum_genes=len(universe),corum_retained_complexes=len(complexes),
                 corum_positive_pairs=len(pos),corum_background_pairs=len(neg),corum_all_pairs=len(all_pairs),
                 string_edges=len(edges),string_background_pairs=len(genes)*(len(genes)-1)//2-len(edges),
                 string_rule='Exact notebook rule: keep protein1 < protein2 and first record per pair; input gene symbols must match')
    io.save_json(result/'pair_universe.json',summary);LOG.info('Pair universe: %s',summary)
    return pos,neg,all_pairs,edges


def pair_scores(z,pairs,index,chunk_size=50000):
    iterator=iter(pairs)
    scores=[]
    while True:
        block=list(islice(iterator,chunk_size))
        if not block:break
        a=np.fromiter((index[g] for g,_ in block),dtype=np.int64,count=len(block))
        b=np.fromiter((index[g] for _,g in block),dtype=np.int64,count=len(block))
        scores.append(np.einsum('ij,ij->i',z[a],z[b]))
    return np.concatenate(scores) if scores else np.empty(0,dtype=z.dtype)


def nonedge_background(z,known_scores):
    """Exact unordered all-pair sum, including the actual norms of zero rows."""
    z=z.astype(np.float64)
    n=len(z);total=n*(n-1)//2;count=total-len(known_scores)
    if count<=0:raise ValueError('No non-STRING pairs available for background')
    all_sum=(np.square(z.sum(axis=0)).sum()-np.square(z).sum())/2
    return float((all_sum-np.asarray(known_scores,dtype=np.float64).sum())/count),count


def compute(profiles,genes,pairs,args,result):
    pospairs,negpairs,allpairs,edges=pairs
    index={g:i for i,g in enumerate(genes)}
    metric_rows=[];corum_frames=[];string_frames=[];evrs={};bin_frames=[];checks={}
    for rep,frame in profiles.items():
        pca=PCA(n_components=args.pca_variance,svd_solver='full')
        with threadpool_limits(limits=args.threads):scores=pca.fit_transform(frame.to_numpy(dtype=np.float32)).astype('float32')
        norms=np.linalg.norm(scores,axis=1,keepdims=True);z=scores/np.maximum(norms,1e-12)
        if not np.isfinite(z).all():raise ValueError('Non-finite normalized PCA profile')
        evrs[rep]=pca.explained_variance_ratio_
        key='model' if rep=='LayerNorm' else 'original'
        pd.DataFrame(z,index=genes).to_parquet(result/key/'hit_PCA_L2_profiles.parquet')
        pos=pair_scores(z,pospairs,index,args.pair_chunk_size);neg=pair_scores(z,negpairs,index,args.pair_chunk_size)
        background=float(neg.mean())
        corum_frames.append(pd.DataFrame(dict(gene1=[a for a,b in pospairs],gene2=[b for a,b in pospairs],representation=rep,
                                              raw_similarity=pos,background_mean=background,background_corrected_similarity=pos-background)))
        raw=pair_scores(z,zip(edges.protein1,edges.protein2),index,args.pair_chunk_size)
        bg,n_bg=nonedge_background(z,raw)
        # Verify the identity on a bounded real-gene subset; the main background still uses every pair.
        verify_n=min(len(genes),256)
        verify_genes=genes[:verify_n];verify_set=set(verify_genes)
        keep=edges.protein1.isin(verify_set)&edges.protein2.isin(verify_set)
        known=set(zip(edges.loc[keep,'protein1'],edges.loc[keep,'protein2']))
        verify_bg,verify_count=nonedge_background(z[:verify_n],raw[keep.to_numpy()])
        direct_sum=0.0;direct_count=0
        for i,g in enumerate(verify_genes):
            js=[j for j in range(i+1,verify_n) if tuple(sorted((g,genes[j]))) not in known]
            if js:
                direct_sum+=float(np.einsum('j,ij->i',z[i].astype(float),z[js].astype(float)).sum());direct_count+=len(js)
        assert direct_count==verify_count
        np.testing.assert_allclose(verify_bg,direct_sum/direct_count,rtol=1e-5,atol=1e-7)
        checks[rep]=dict(nonedge_formula_verified=True,verification_genes=verify_n,n_background=n_bg,zero_norm_rows=int((norms[:,0]<1e-12).sum()))
        sf=edges.copy();sf['representation']=rep;sf['raw_similarity']=raw
        sf['background_mean']=bg;sf['background_corrected_similarity']=raw-bg
        try:sf['similarity_bin']=pd.qcut(raw,q=5,labels=np.arange(1,6)).astype(int)
        except ValueError as exc:raise ValueError('Cannot form five distinct similarity quantiles; inspect tied scores') from exc
        string_frames.append(sf)
        bins=sf.groupby('similarity_bin',observed=True).combined_score.agg(count='size',mean='mean',median='median',sem='sem').reset_index()
        bins['representation']=rep;bin_frames.append(bins)
        rho,rho_p=spearmanr(raw,sf.combined_score)
        auc_mask=(sf.combined_score<=args.string_low_score)|(sf.combined_score>=args.string_high_score)
        classes=(sf.loc[auc_mask,'combined_score']>=args.string_high_score).astype(int)
        auc=roc_auc_score(classes,sf.loc[auc_mask,'raw_similarity']) if classes.nunique()==2 else np.nan
        high_rate=float((sf.combined_score>=args.string_high_score).mean())
        top=sf.similarity_bin==5;top_rate=float((sf.loc[top,'combined_score']>=args.string_high_score).mean())
        metric_rows.append(dict(strategy='hit_then_PCA' if args.gene_selection_mode=='hit_list' else 'all_genes_then_PCA',representation=rep,pca_fit_genes=len(genes),pca_components=pca.n_components_,
           pca_explained_variance=float(evrs[rep].sum()),corum_positive_mean=float(pos.mean()),
           corum_all_pair_mean=float(pair_scores(z,allpairs,index,args.pair_chunk_size).mean()),corum_background_mean=background,
           corum_delta=float(pos.mean()-background),corum_AUROC=float(roc_auc_score(np.r_[np.ones(len(pos)),np.zeros(len(neg))],np.r_[pos,neg])),
           string_edges=len(sf),non_STRING_background_pairs=n_bg,string_background_mean=bg,string_spearman_rho=float(rho),string_spearman_p=float(rho_p),
           string_high_vs_low_AUROC=float(auc),string_top_bin_mean=float(sf.loc[top,'combined_score'].mean()),
           string_top_bin_high_conf_rate=top_rate,string_top_bin_high_conf_enrichment=top_rate/high_rate if high_rate else np.nan))
    metrics=pd.DataFrame(metric_rows);corum=pd.concat(corum_frames,ignore_index=True);string=pd.concat(string_frames,ignore_index=True);bins=pd.concat(bin_frames,ignore_index=True)
    metrics.to_csv(result/'metrics.csv',index=False);corum.to_csv(result/'CORUM_background_corrected.csv',index=False)
    string.to_csv(result/'STRING_background_corrected.csv',index=False);bins.to_csv(result/'STRING_similarity_quantiles_summary.csv',index=False)
    io.save_json(result/'validation.json',checks)
    return metrics,corum,string,bins,evrs


def export(fig,folder,stem,args):
    io.export(fig,folder/stem,args.png_dpi,formats=args.figure_formats)


def similarity_plot_style():
    """Shared Arial typography for the CORUM and STRING similarity figures."""
    from matplotlib import font_manager
    font_dir = PROJECT/'tutorials/09_paper_plot_code/code/fonts/arial'
    for font in sorted(font_dir.glob('*.TTF')):
        font_manager.fontManager.addfont(str(font))
    font_manager.findfont('Arial', fallback_to_default=False)
    return {
        'font.family': 'Arial',
        'font.sans-serif': ['Arial'],
        'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 16,
        'xtick.labelsize': 11, 'ytick.labelsize': 11,
        'legend.fontsize': 12, 'axes.linewidth': 1.2,
        'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
        'xtick.major.size': 6, 'ytick.major.size': 6,
        'pdf.fonttype': 42, 'svg.fonttype': 'none',
    }


def plot_corum_distribution(corum, args, folder):
    """Preserve the reference notebook's CORUM histogram style."""
    plot_colors = {'LayerNorm': '#BC3C28', 'Original': '#0072B5'}
    plot_labels = {'LayerNorm': 'DeSCORT', 'Original': 'Original'}
    style = similarity_plot_style()
    with plt.rc_context():
        sns.set_theme(style='ticks', rc=style)
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        # Draw Original first so the red DeSCORT histogram and KDE sit on top.
        for layer, rep in enumerate(['Original', 'LayerNorm']):
            values = corum.loc[corum.representation == rep,
                               'background_corrected_similarity'].to_numpy()
            sns.histplot(
                values, color=plot_colors[rep], stat='percent', bins=80, kde=True,
                line_kws={'lw': 1.8, 'zorder': 3 + 2 * layer}, alpha=.35, ax=ax,
                edgecolor='white', linewidth=.5,
                label=plot_labels[rep], zorder=2 + 2 * layer,
            )
            ax.axvline(values.mean(), color=plot_colors[rep], linestyle='--', linewidth=1.8,
                       zorder=3 + 2 * layer)
        ax.axvline(0, color='.35', linewidth=.8, linestyle=':')
        ax.set_ylabel('Gene-pair frequency (%)')
        ax.set_xlabel('')
        ax.set_title('CORUM co-complex similarity')
        handles, labels = ax.get_legend_handles_labels()
        order = [labels.index(name) for name in ['DeSCORT', 'Original']]
        ax.legend([handles[i] for i in order], [labels[i] for i in order],
                  frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(.5, -.10),
                  borderaxespad=.2, handletextpad=.5, columnspacing=1.6)
        sns.despine(ax=ax)
        # Keep axes above both histogram layers, including the bars' white edges.
        for side in ['bottom', 'left']:
            ax.spines[side].set_zorder(10)
        ax.xaxis.set_zorder(11)
        ax.yaxis.set_zorder(11)
        fig.tight_layout()
        export(fig, folder, 'CORUM_background_corrected_distribution', args)


def plot_string_similarity_line(string, args, folder):
    """Summarize STRING scores in fixed background-corrected similarity intervals."""
    edges = np.array([-1., -.6, -.2, .2, .6, 1.])
    labels = ['-1 ~ -0.6', '-0.6 ~ -0.2', '-0.2 ~ 0.2', '0.2 ~ 0.6', '0.6 ~ 1']
    records = []
    for rep in ['LayerNorm', 'Original']:
        frame = string.loc[string.representation.eq(rep)].copy()
        values = frame['background_corrected_similarity'].to_numpy(float)
        if not len(frame) or not np.isfinite(values).all():
            raise ValueError(f'{rep}: missing or nonfinite STRING similarity values')
        frame['fixed_interval'] = pd.cut(values, bins=edges, labels=range(1, 6),
                                          include_lowest=True, right=True)
        if frame.fixed_interval.isna().any():
            raise ValueError(f'{rep}: corrected similarities outside [-1, 1]; fixed bins would omit pairs')
        stats = frame.groupby('fixed_interval', observed=True).combined_score.agg(
            count='size', mean='mean', median='median', sem='sem').reindex(range(1, 6))
        stats['count'] = stats['count'].fillna(0).astype(int)
        assert stats['count'].sum() == len(frame)
        for i, row in stats.iterrows():
            records.append(dict(representation=rep, similarity_bin=i, interval=labels[i-1],
                                lower_bound=edges[i-1], upper_bound=edges[i],
                                count=int(row['count']), mean=row['mean'],
                                median=row['median'], sem=row['sem']))
    summary = pd.DataFrame(records)
    summary.to_csv(folder.parent/'results/STRING_similarity_intervals_summary.csv', index=False)
    colors = {'LayerNorm': '#BC3C28', 'Original': '#0072B5'}
    names = {'LayerNorm': 'DeSCORT', 'Original': 'Original'}
    with plt.rc_context():
        sns.set_theme(style='ticks', rc=similarity_plot_style())
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        centers = (edges[:-1]+edges[1:])/2
        for layer, rep in enumerate(['Original', 'LayerNorm']):
            sub = summary[summary.representation.eq(rep)].sort_values('similarity_bin')
            ax.errorbar(centers, sub['mean'], yerr=1.96*sub['sem'],
                        color=colors[rep], marker='o', lw=2, capsize=3,
                        label=names[rep], zorder=3+2*layer)
        ax.set(xticks=centers, xticklabels=labels, xlim=(-1, 1),
               xlabel='Cosine similarity',
               ylabel='Mean STRING combined score', title='STRING interaction confidence')
        handles, legend_labels = ax.get_legend_handles_labels()
        order = [legend_labels.index(name) for name in ['DeSCORT', 'Original']]
        ax.legend([handles[i] for i in order], [legend_labels[i] for i in order],
                  frameon=False, loc='upper left')
        sns.despine(ax=ax)
        for side in ['bottom', 'left']:
            ax.spines[side].set_zorder(10)
        ax.xaxis.set_zorder(11)
        ax.yaxis.set_zorder(11)
        fig.tight_layout()
        # Keep the established figure filename; the new CSV records actual fixed bins.
        export(fig, folder, 'STRING_similarity_quantiles_line', args)
    return summary


def plot_results(corum,string,bins,evrs,args,folder):
    selection='hit' if args.gene_selection_mode=='hit_list' else 'all common genes'
    title=f'{args.dataset_name} ({selection} → PCA)'
    plot_corum_distribution(corum, args, folder)
    # Fixed confidence bins make subtraction visible on the similarity axis.
    sf=string.copy();sf['model']=sf.representation.map(DISPLAY)
    palette={DISPLAY[k]:COLORS[k] for k in COLORS}
    fig,ax=plt.subplots(figsize=(8.2,4.8))
    sns.boxenplot(data=sf,x='confidence_interval',y='background_corrected_similarity',hue='model',order=STRING_LABELS,
                  hue_order=list(DISPLAY.values()),palette=palette,showfliers=False,linewidth=.65,ax=ax)
    ax.axhline(0,color='.35',linestyle=':',lw=.8)
    ax.set(xlabel='STRING confidence interval (combined score / 1000)',ylabel='Cosine similarity − non-STRING background mean',title=f'{title}: STRING confidence vs corrected similarity')
    ax.legend(title=None,frameon=False,loc='upper center',bbox_to_anchor=(.5,-.17),ncol=2)
    fig.tight_layout();export(fig,folder,'STRING_confidence_background_corrected',args)
    # Subtracting a representation-wide constant cannot change these quantile ranks.
    fig,ax=plt.subplots(figsize=(8.2,4.5))
    sns.boxenplot(data=sf,x='similarity_bin',y='combined_score',hue='model',order=[1,2,3,4,5],
                  hue_order=list(DISPLAY.values()),palette=palette,showfliers=False,linewidth=.65,ax=ax)
    ax.set(xticks=range(5),xticklabels=['Q1','Q2','Q3','Q4','Q5'],ylim=(0,1000),xlabel='Corrected-similarity quantile (low → high)',ylabel='STRING combined score',title=f'{title}: STRING score by similarity quantile')
    ax.legend(title=None,frameon=False,loc='upper center',bbox_to_anchor=(.5,-.17),ncol=2)
    fig.tight_layout();export(fig,folder,'STRING_similarity_quantiles_boxen',args)
    plot_string_similarity_line(string, args, folder)
    fig,ax=plt.subplots(figsize=(5.8,4))
    for rep in COLORS:ax.plot(range(1,len(evrs[rep])+1),np.cumsum(evrs[rep])*100,color=COLORS[rep],lw=2,label=DISPLAY[rep])
    ax.set(xlabel='Number of principal components',ylabel='Cumulative explained variance (%)',title=f'{title}: PCA explained variance')
    ax.legend(frameon=False);fig.tight_layout();export(fig,folder,'PCA_explained_variance',args)


def paired_changes(metrics,corum,string,args,result,folder):
    summary={}
    for name,frame,keys in [('CORUM',corum,['gene1','gene2']),('STRING',string,['protein1','protein2'])]:
        model=frame[frame.representation=='LayerNorm'].copy();original=frame[frame.representation=='Original'].copy()
        pair=model.merge(original,on=keys,suffixes=('_model','_original'),validate='one_to_one')
        assert len(pair)==len(model)==len(original)
        out=pair[keys].copy();out['raw_model_minus_original']=pair.raw_similarity_model-pair.raw_similarity_original
        out['corrected_model_minus_original']=pair.background_corrected_similarity_model-pair.background_corrected_similarity_original
        if name=='STRING':out['combined_score']=pair.combined_score_model;out['similarity_bin']=pair.similarity_bin_model
        out.to_csv(result/f'{name}_paired_model_minus_original.csv',index=False)
        values=out.corrected_model_minus_original
        summary[name+'_corrected_mean_change']=float(values.mean());summary[name+'_pairs_improved_fraction']=float((values>0).mean())
        fig,ax=plt.subplots(figsize=(7,4.2))
        if name=='CORUM':
            sns.histplot(values,bins=70,kde=True,stat='percent',color=COLORS['LayerNorm'],alpha=.45,ax=ax)
            ax.axvline(0,color='.35',linestyle=':');ax.axvline(values.mean(),color=COLORS['LayerNorm'],linestyle='--',label=f'Mean {values.mean():.3f}')
            ax.set(xlabel='Corrected CORUM similarity: model − original',ylabel='Pair frequency (%)',title=f'{args.dataset_name}: paired CORUM change');ax.legend(frameon=False)
        else:
            stats=out.groupby('similarity_bin').corrected_model_minus_original.agg(count='size',mean='mean',median='median').reset_index()
            stats.to_csv(result/'STRING_paired_change_by_model_quantile.csv',index=False)
            ax.plot(stats.similarity_bin,stats['mean'],color=COLORS['LayerNorm'],marker='o');ax.axhline(0,color='.35',linestyle=':')
            ax.set(xticks=range(1,6),xticklabels=['Q1','Q2','Q3','Q4','Q5'],xlabel='Model corrected-similarity quantile (low → high)',ylabel='Mean corrected similarity change',title=f'{args.dataset_name}: paired STRING change')
        fig.tight_layout();export(fig,folder,f'{name}_paired_model_minus_original',args)
    by=metrics.set_index('representation')
    for name,col in [('CORUM_AUROC','corum_AUROC'),('CORUM_delta','corum_delta'),('STRING_rho','string_spearman_rho'),('STRING_AUROC','string_high_vs_low_AUROC')]:
        summary[name+'_model_minus_original']=float(by.loc['LayerNorm',col]-by.loc['Original',col])
    pd.DataFrame([summary]).to_csv(result/'paired_change_summary.csv',index=False)
