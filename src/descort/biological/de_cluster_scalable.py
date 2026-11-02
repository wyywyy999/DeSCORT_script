"""Memory-bounded exact cosine operations for the all-gene cluster workflow."""
from pathlib import Path
import hashlib,json
import numpy as np
from sklearn.metrics import roc_auc_score


def pair_scores(unit,first,second,chunk_size=50000):
    first=np.asarray(first,dtype=np.int64);second=np.asarray(second,dtype=np.int64)
    scores=np.empty(len(first),dtype=unit.dtype)
    for start in range(0,len(first),chunk_size):
        stop=min(start+chunk_size,len(first))
        scores[start:stop]=np.einsum('ij,ij->i',unit[first[start:stop]],unit[second[start:stop]])
    return scores


def topk(unit,k,block_size=256):
    n=len(unit)
    if not 0<k<n:raise ValueError('Neighbour count must be smaller than the gene count')
    indices=np.empty((n,k),dtype=np.int64);scores=np.empty((n,k),dtype=unit.dtype)
    for start in range(0,n,block_size):
        stop=min(start+block_size,n)
        sim=unit[start:stop]@unit.T
        sim[np.arange(stop-start),np.arange(start,stop)]=-np.inf
        selected=np.argpartition(-sim,kth=np.arange(k),axis=1)[:,:k]
        values=np.take_along_axis(sim,selected,axis=1);order=np.argsort(-values,axis=1)
        indices[start:stop]=np.take_along_axis(selected,order,axis=1)
        scores[start:stop]=np.take_along_axis(values,order,axis=1)
    return indices,scores


def mean_all_pairs(unit):
    x=unit.astype(np.float64)
    return float((np.square(x.sum(0)).sum()-np.square(x).sum())/(len(x)*(len(x)-1)))


def all_pair_pool(unit,folder,name,block_size=256):
    """Exact upper triangle in legacy row-major order, stored as a float32 memmap."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    n=len(unit);total=n*(n-1)//2
    info=dict(genes=n,shape=list(unit.shape),dtype=str(unit.dtype),
              sha256=hashlib.sha256(np.ascontiguousarray(unit).tobytes()).hexdigest())
    path=folder/(name+'.f32');meta=folder/(name+'.json')
    if path.exists() and meta.exists() and json.loads(meta.read_text())==info and path.stat().st_size==total*4:
        return np.memmap(path,mode='r',dtype='float32',shape=(total,))
    temp=folder/(name+'.partial')
    out=np.memmap(temp,mode='w+',dtype='float32',shape=(total,));offset=0
    for start in range(0,n,block_size):
        stop=min(start+block_size,n);sim=unit[start:stop]@unit.T
        for local,i in enumerate(range(start,stop)):
            width=n-i-1
            out[offset:offset+width]=sim[local,i+1:];offset+=width
    assert offset==total
    out.flush();del out;temp.replace(path);meta.write_text(json.dumps(info,indent=2)+'\n')
    return np.memmap(path,mode='r',dtype='float32',shape=(total,))


def resampled_means(pool,rng,repeats,pair_count,max_elements=1000000):
    result=np.empty(repeats,dtype=np.float64)
    batch=max(1,max_elements//pair_count)
    for start in range(0,repeats,batch):
        stop=min(repeats,start+batch)
        idx=rng.integers(0,len(pool),(stop-start,pair_count))
        result[start:stop]=pool[idx].mean(axis=1)
    return result


def sample_nonedges(n,positive_codes,count,rng):
    """Uniform sampling without replacement over unordered nonedges; no full universe."""
    selected=[];seen=set()
    while len(selected)<count:
        size=max(256,2*(count-len(selected)))
        first=rng.integers(0,n,size);second=rng.integers(0,n,size)
        for a,b in zip(first,second):
            if a==b:continue
            code=int(min(a,b)*n+max(a,b))
            if code in positive_codes or code in seen:continue
            seen.add(code);selected.append(code)
            if len(selected)==count:break
    values=np.asarray(selected,dtype=np.int64)
    return values//n,values%n


def evaluate_support(model,original,pair_sets,model_nn,original_nn,top_k,repeats,seed,
                     empirical_p,chunk_size=50000):
    n=len(model);total=n*(n-1)//2
    def top_codes(nn):
        first=np.repeat(np.arange(n),top_k);second=nn[:,:top_k].ravel()
        return set((np.minimum(first,second)*n+np.maximum(first,second)).tolist())
    model_top=top_codes(model_nn);original_top=top_codes(original_nn)
    model_mean=mean_all_pairs(model);original_mean=mean_all_pairs(original)
    rng=np.random.default_rng(seed);rows=[]
    for i,(db,codes) in enumerate(pair_sets.items()):
        valid={int(c) for c in codes if 0<=c//n<c%n<n}
        if not valid:continue
        first=np.fromiter(valid,dtype=np.int64);second=first%n;first=first//n
        mp=pair_scores(model,first,second,chunk_size);op=pair_scores(original,first,second,chunk_size)
        sample_n=min(len(valid),total-len(valid),200000)
        if not sample_n:raise ValueError(f'{db}: no negative pairs available for AUROC')
        a,b=sample_nonedges(n,valid,sample_n,rng)
        mn=pair_scores(model,a,b,chunk_size);on=pair_scores(original,a,b,chunk_size)
        prevalence=len(valid)/total;mr=len(model_top&valid)/max(1,len(model_top));orr=len(original_top&valid)/max(1,len(original_top))
        rows.append(dict(database=db,positive_pairs=len(valid),all_pair_prevalence=prevalence,
             model_positive_mean_cosine=float(mp.mean()),original_positive_mean_cosine=float(op.mean()),
             model_all_pair_mean_cosine=model_mean,original_all_pair_mean_cosine=original_mean,
             model_delta_vs_all_pairs=float(mp.mean()-model_mean),original_delta_vs_all_pairs=float(op.mean()-original_mean),
             model_AUROC=float(roc_auc_score(np.r_[np.ones(len(mp)),np.zeros(sample_n)],np.r_[mp,mn])),
             original_AUROC=float(roc_auc_score(np.r_[np.ones(len(op)),np.zeros(sample_n)],np.r_[op,on])),
             model_top20_edge_rate=mr,original_top20_edge_rate=orr,model_top20_fold=mr/prevalence,original_top20_fold=orr/prevalence,
             model_permutation_p=empirical_p(model,first,second,float(mp.mean()),repeats,seed+i),
             original_permutation_p=empirical_p(original,first,second,float(op.mean()),repeats,seed+100+i)))
    return rows
