"""Checkpoint-only, resumable sharded inference: one independent process per visible GPU.

GT and homographies are never passed to the detector. Model outputs are in input-image pixels.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(s) for s in f if s.strip()]


def json_line(obj):
    return json.dumps(obj,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n'


def build_tasks(manifest,locations,smoke_sources=0):
    rows=read_jsonl(manifest)
    if smoke_sources:
        # Deterministic sample spanning perspective and identity; IDs remain metadata-derived.
        source_groups={}
        for r in rows:
            source_groups.setdefault(r['source_image_id'],[]).append(r)
        geom=sorted(s for s,rs in source_groups.items() if any(r['has_perspective'] for r in rs))
        plain=sorted(set(source_groups)-set(geom))
        chosen=[]
        while len(chosen)<smoke_sources and (geom or plain):
            if geom:chosen.append(geom.pop(0))
            if plain and len(chosen)<smoke_sources:chosen.append(plain.pop(0))
        rows=[r for r in rows if r['source_image_id'] in set(chosen)]
    unique={r['source_image_id']:r for r in rows}
    tasks=[]
    for source,r in sorted(unique.items()):
        tasks.append(dict(sample_id='clean::'+source,source_image_id=source,condition='clean',
                          path=str(Path(locations['clean_root'])/r['clean_relpath']),
                          image_sha256=r['clean_sha256']))
    for r in rows:
        tasks.append(dict(sample_id='deg::'+r['variant_id'],source_image_id=r['source_image_id'],
                          variant_id=r['variant_id'],condition='degraded',
                          path=str(Path(locations['degraded_root'])/r['degraded_relpath']),
                          image_sha256=r['degraded_sha256']))
    if len({t['sample_id'] for t in tasks})!=len(tasks):raise ValueError('Duplicate task IDs')
    return tasks


def resume_rows(path):
    """Repair only an interrupted final JSONL fragment, saving that fragment as evidence."""
    path=Path(path)
    if not path.exists():return []
    data=path.read_bytes();lines=data.splitlines(keepends=True);good=[];offset=0;repaired=False
    for i,line in enumerate(lines):
        try:
            item=json.loads(line)
        except json.JSONDecodeError:
            if i!=len(lines)-1:raise ValueError(f'Corrupted middle of {path}; do not silently drop predictions')
            path.with_suffix(path.suffix+'.interrupted_tail').write_bytes(line)
            with path.open('r+b') as f:f.truncate(offset)
            repaired=True
            print('Recovered interrupted final line:',path,flush=True)
            break
        good.append(item);offset+=len(line)
    if good and data and not repaired and not data.endswith(b'\n'):
        with path.open('ab') as f:f.write(b'\n')
    return good


def worker(a):
    import math
    import torch
    from mmengine.config import Config
    from mmengine.runner import load_checkpoint
    from mmengine.utils import import_modules_from_strings
    from mmocr.apis import TextDetInferencer
    from mmocr.utils import register_all_modules

    tasks=read_jsonl(a.tasks)
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    fingerprint=dict(manifest_sha256=a.manifest_hash,checkpoint_sha256=sha256(a.checkpoint),
                     config_sha256=sha256(a.config),tasks_sha256=sha256(a.tasks),model='monkey',
                     worker_source_sha256=sha256(__file__),torch_version=torch.__version__,
                     gpu_name=torch.cuda.get_device_name(0))
    if fingerprint['checkpoint_sha256']!=a.checkpoint_hash or fingerprint['config_sha256']!=a.config_hash:
        raise ValueError('Checkpoint/config differ from verified Phase 1 artifacts')
    meta=out/'run_identity.json'
    if meta.exists() and json.loads(meta.read_text())!=fingerprint:
        raise ValueError('Existing predictions use a different model/config/manifest/shard. Use a new run directory.')
    meta.write_text(json.dumps(fingerprint,indent=2)+'\n')
    pred_path=out/'predictions.jsonl'
    existing=resume_rows(pred_path)
    task_ids={t['sample_id'] for t in tasks}
    seen={r['sample_id'] for r in existing}
    if len(seen)!=len(existing) or not seen<=task_ids:
        raise ValueError('Duplicate/unexpected resume predictions')
    if any(r['manifest_sha256']!=a.manifest_hash or r['checkpoint_sha256']!=a.checkpoint_hash for r in existing):
        raise ValueError('Resume output model/manifest identity mismatch')
    todo=[t for t in tasks if t['sample_id'] not in seen]
    if not todo:
        print('Shard complete; no new inference:',len(tasks),flush=True)
        return
    assert torch.cuda.is_available() and torch.cuda.device_count()==1,'Worker must see exactly one assigned GPU'
    cfg=Config.fromfile(a.config)
    if cfg.get('custom_imports'):import_modules_from_strings(**cfg.custom_imports)
    register_all_modules(init_default_scope=True)
    detector=TextDetInferencer(model=a.config,weights=a.checkpoint,device='cuda:0')
    detector.model.eval()
    # Strict-load once more: reject missing random heads rather than accepting a partial checkpoint.
    load_checkpoint(detector.model,a.checkpoint,map_location='cpu',strict=True)
    protocol=dict(physical_visible_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  gpu_name=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,
                  model=cfg.model,test_pipeline=cfg.test_dataloader.dataset.pipeline,
                  effective_inferencer_pipeline=repr(detector.pipeline),
                  score_policy='raw detector output; no extra score filter/sweep')
    (out/'inference_protocol.json').write_text(json.dumps(protocol,indent=2,default=str)+'\n')
    def predict(path):
        with torch.inference_mode():
            result=detector(path,batch_size=1,return_vis=False,save_vis=False,save_pred=False,show=False)
        assert len(result['predictions'])==1
        return result['predictions'][0]
    # Warmup uses only pixels, never GT, and does not add duplicated saved predictions.
    for _ in range(2):predict(todo[0]['path'])
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter();new_count=0
    with pred_path.open('a',encoding='utf-8') as f:
        for task in todo:
            task_start=time.perf_counter()
            if sha256(task['path'])!=task['image_sha256']:
                raise ValueError(f'Input image bytes changed after audit: {task["sample_id"]}')
            torch.cuda.synchronize();t=time.perf_counter()
            pred=predict(task['path'])
            torch.cuda.synchronize();elapsed=(time.perf_counter()-t)*1000
            polygons=[[float(v) for v in p] for p in pred['polygons']]
            scores=[float(v) for v in pred['scores']]
            assert len(polygons)==len(scores)
            assert all(math.isfinite(v) for p in polygons for v in p)
            assert all(math.isfinite(v) and 0<=v<=1 for v in scores)
            row={k:v for k,v in task.items() if k not in {'path','image_sha256'}}
            row.update(polygons=polygons,scores=scores,model='monkey',manifest_sha256=a.manifest_hash,
                       checkpoint_sha256=a.checkpoint_hash,config_sha256=a.config_hash,
                       inference_ms_with_decode_and_postprocess=elapsed,
                       processing_ms_with_hash_and_decode=(time.perf_counter()-task_start)*1000)
            f.write(json_line(row));f.flush();new_count+=1
            if new_count%25==0 or new_count==len(todo):
                print(f'PROGRESS {len(seen)+new_count}/{len(tasks)} | new elapsed {time.perf_counter()-start:.1f}s',flush=True)
    (out/'worker_summary.json').write_text(json.dumps(dict(new_images=new_count,resumed_images=len(seen),
         wall_seconds_new=time.perf_counter()-start,max_allocated_gpu_mib=torch.cuda.max_memory_allocated()/1024**2),indent=2)+'\n')


def launch(a):
    locations=json.loads(Path(a.locations).read_text())
    tasks=build_tasks(a.manifest,locations,a.smoke_sources)
    manifest_hash=sha256(a.manifest)
    root=Path(a.output_dir);root.mkdir(parents=True,exist_ok=True)
    gpu_ids=[x.strip() for x in a.gpus.split(',') if x.strip()]
    if not gpu_ids or len(gpu_ids)!=len(set(gpu_ids)):raise ValueError('Set one or more unique GPU IDs')
    processes=[];handles=[];started=time.perf_counter()
    try:
        for shard,gpu in enumerate(gpu_ids):
            subset=tasks[shard::len(gpu_ids)]
            task_file=root/f'tasks_gpu{gpu}.jsonl'
            content=''.join(json_line(t) for t in subset)
            if task_file.exists() and task_file.read_text()!=content:
                raise ValueError('Sharding changed; use a new output directory instead of mixing runs')
            task_file.write_text(content,encoding='utf-8')
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu
            cmd=[sys.executable,str(Path(__file__).resolve()),'--worker','--tasks',str(task_file),
                 '--output-dir',str(root/f'gpu{gpu}'),'--manifest-hash',manifest_hash,
                 '--config',a.config,'--checkpoint',a.checkpoint,
                 '--config-hash',a.config_hash,'--checkpoint-hash',a.checkpoint_hash]
            handle=(root/f'gpu{gpu}.log').open('w',encoding='utf-8');handles.append(handle)
            processes.append((gpu,subprocess.Popen(cmd,env=env,stdout=handle,stderr=subprocess.STDOUT)))
            print('Started GPU',gpu,'tasks',len(subset),flush=True)
        last=0
        while any(p.poll() is None for _,p in processes):
            failures=[(g,p.returncode) for g,p in processes if p.poll() not in (None,0)]
            if failures:raise RuntimeError(f'Inference worker failed {failures}; see per-GPU logs. Partial outputs are NOT scored.')
            if time.perf_counter()-last>=30:
                last=time.perf_counter()
                progress={g:sum(1 for _ in (root/f'gpu{g}/predictions.jsonl').open(encoding='utf-8'))
                          if (root/f'gpu{g}/predictions.jsonl').exists() else 0 for g,_ in processes}
                print('SHARD PROGRESS',progress,'elapsed_s',round(time.perf_counter()-started),flush=True)
            time.sleep(1)
        if any(p.returncode for _,p in processes):raise RuntimeError('Worker failed; see logs')
    finally:
        for _,p in processes:
            if p.poll() is None:p.terminate()
        for _,p in processes:
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill();p.wait(timeout=15)
        for h in handles:h.close()
    merged={}
    for gpu in gpu_ids:
        for row in read_jsonl(root/f'gpu{gpu}/predictions.jsonl'):
            key=row['sample_id']
            if key in merged:raise ValueError('Duplicate merged prediction: '+key)
            merged[key]=row
    if set(merged)!={t['sample_id'] for t in tasks}:
        raise ValueError('Incomplete or extra merged predictions; not evaluated')
    merged_path=root/'monkey_predictions.jsonl'
    merged_path.write_text(''.join(json_line(merged[k]) for k in sorted(merged)),encoding='utf-8')
    per_gpu={}
    for gpu in gpu_ids:
        values=[r['processing_ms_with_hash_and_decode'] for r in read_jsonl(root/f'gpu{gpu}/predictions.jsonl')]
        per_gpu[gpu]=dict(samples=len(values),mean_ms=sum(values)/len(values) if values else 0.)
    full_count=len(build_tasks(a.manifest,locations))
    eta=max((v['mean_ms']*(full_count//len(gpu_ids)+1)/3_600_000 for v in per_gpu.values()),default=0.)
    summary=dict(images=len(tasks),unique_sources=len({t['source_image_id'] for t in tasks}),
                 manifest_sha256=manifest_hash,predictions_sha256=sha256(merged_path),
                 per_gpu=per_gpu,wall_seconds_this_launch=time.perf_counter()-started,
                 estimated_full_inference_hours_excludes_setup_and_evaluation=eta,
                 note='ETA extrapolation from actual inputs, not a guarantee; resume totals include prior images')
    (root/'inference_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true')
    p.add_argument('--manifest');p.add_argument('--locations');p.add_argument('--gpus',default='0,1')
    p.add_argument('--smoke-sources',type=int,default=0);p.add_argument('--output-dir',required=True)
    p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--config-hash',required=True);p.add_argument('--checkpoint-hash',required=True)
    p.add_argument('--tasks');p.add_argument('--manifest-hash')
    a=p.parse_args()
    worker(a) if a.worker else launch(a)
