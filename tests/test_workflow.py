"""CPU fixtures only; these tests do not create or approve research results."""
import copy
import gzip
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image

from cs431 import gate, icdar, inference, manifest, workflow
from cs431.common import ROOT, canonical, identity, jsonl, read_json, run_lock, save_json, sha256, write_jsonl

QUAD = [0., 0., 10., 0., 10., 10., 0., 10.]


def test_temp():
    # Keep all fixture writes/atomic renames inside the authorized workspace.
    parent=ROOT/'outputs/cpu_tests'
    parent.mkdir(parents=True,exist_ok=True)
    return tempfile.TemporaryDirectory(dir=parent)


def evaluated_fixture(root):
    """Synthetic complete evidence solely for testing receipt integrity in a temp dir."""
    run = root/'test_run'
    run.mkdir()
    cid = gate.clone_id(root)
    config = dict(test_fixture=True)
    save_json(run/'config_snapshot.json', config)
    model = dict(config_sha256=sha256(run/'config_snapshot.json'), config_content_sha256=identity(config))
    save_json(run/'model_identity.json', model)
    records = [dict(image_id=f'img_{i}', image_sha256='a'*64, gt_sha256='b'*64) for i in range(1,501)]
    inventory = dict(benchmark='ICDAR2015 test', images=500, records=records, dataset_identity=identity(records))
    save_json(run/'dataset_inventory.json', inventory)
    run_hash = identity(dict(dataset=inventory, clone_id=cid))
    write_jsonl(run/'inference/predictions.jsonl', [dict(sample_id=r['image_id'], model='paddle',
        manifest_sha256=run_hash, model_identity_sha256=identity(model), polygons=[QUAD], scores=[.9]) for r in records])
    write_jsonl(run/'per_image.jsonl', [dict(sample_id=r['image_id'], gt=1, predictions=1, matched=1) for r in records])
    save_json(run/'metrics.json', dict(images=500, measured=dict(precision=1.,recall=1.,hmean=1.),
        counts=dict(gt=500,predictions=500,matched=500), exact_paddle_83_8_reproduction='NOT VERIFIED'))
    for name in ('pip_freeze.txt','REVIEW_ME.md'):
        (run/name).write_text('SYNTHETIC UNIT TEST ONLY')
    (run/'overlays').mkdir()
    for i in range(20):
        (run/f'overlays/{i}.jpg').write_bytes(b'unit-test hash fixture, not a real overlay')
    report = dict(status='EVALUATED_REQUIRES_REVIEW', clone_id=cid, benchmark='ICDAR2015 test',
                  images=500, model_identity=model,
                  artifacts={p.relative_to(run).as_posix():sha256(p) for p in run.rglob('*') if p.is_file()})
    save_json(run/'phase1_result.json', report)
    return run, model


class GateTests(unittest.TestCase):
    def setUp(self):
        self.temp = test_temp()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_fresh_clone_blocks_without_gpu_packages(self):
        with self.assertRaisesRegex(RuntimeError,'PHASE 2 BLOCKED'):
            gate.require_phase1(root=self.root)

    def test_evaluation_alone_does_not_unlock(self):
        evaluated_fixture(self.root)
        with self.assertRaisesRegex(RuntimeError,'BLOCKED'):
            gate.require_phase1(root=self.root)

    def test_review_and_current_identity_then_mutation_rejected(self):
        run, model = evaluated_fixture(self.root)
        gate.approve(run,'Test reviewer','Synthetic test fixture; exercises the local gate only.',self.root)
        self.assertEqual(gate.require_phase1(model,self.root)['reviewer'],'Test reviewer')
        with self.assertRaisesRegex(RuntimeError,'changed since Phase 1'):
            gate.require_phase1(dict(model,another_version=True),self.root)
        (run/'overlays/0.jpg').write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError,'artifact changed'):
            gate.require_phase1(model,self.root)

    def test_receipt_copied_to_another_clone_fails(self):
        run, model = evaluated_fixture(self.root)
        gate.approve(run,'Test','Synthetic test reviewer assessment of the fixture.',self.root)
        other = self.root/'other_clone'
        gate.clone_id(other)
        save_json(other/'.local/paddle_phase1_review.json',read_json(self.root/'.local/paddle_phase1_review.json'))
        with self.assertRaisesRegex(RuntimeError,'another clone'):
            gate.require_phase1(model,other)

    def test_full_evidence_and_non_degenerate_counts_required(self):
        run, _ = evaluated_fixture(self.root)
        report = read_json(run/'phase1_result.json')
        report['artifacts'].pop('inference/predictions.jsonl')
        save_json(run/'phase1_result.json', report)
        with self.assertRaisesRegex(RuntimeError,'Missing required'):
            gate.approve(run,'Test','Synthetic test fixture requiring complete evidence.',self.root)

    def test_review_notes_required(self):
        run,_ = evaluated_fixture(self.root)
        with self.assertRaises(ValueError):
            gate.approve(run,'Test','ok',self.root)

    def test_metric_counts_cannot_disagree(self):
        run,_ = evaluated_fixture(self.root)
        m = read_json(run/'metrics.json');m['counts']['matched']=250
        save_json(run/'metrics.json',m)
        report=read_json(run/'phase1_result.json');report['artifacts']['metrics.json']=sha256(run/'metrics.json')
        save_json(run/'phase1_result.json',report)
        with self.assertRaisesRegex(RuntimeError,'raw counts'):
            gate.verify_result(run,self.root)

    def test_controller_lock_refuses_concurrent_writer_and_cleans_up(self):
        with run_lock(self.root/'out'):
            with self.assertRaisesRegex(RuntimeError,'locked'):
                with run_lock(self.root/'out'):
                    self.fail('second controller entered')
        self.assertFalse((self.root/'out/.controller.lock').exists())


class DataAndInferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=test_temp();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def test_icdar_parser_handles_bom_commas_and_ignored_text(self):
        p=self.root/'gt.txt'
        p.write_text('\ufeff0,0,10,0,10,10,0,10,hello,world\n0,0,2,0,2,2,0,2,###\n',encoding='utf-8')
        rows=icdar.parse_gt(p)
        self.assertEqual([r['ignore'] for r in rows],[False,True])
        p.write_text('0,0,1,1,bad\n')
        with self.assertRaises(ValueError):icdar.parse_gt(p)

    def test_phase1_rejects_partial_dataset(self):
        with self.assertRaisesRegex(ValueError,'exactly ICDAR2015'):
            icdar.dataset(self.root,self.root)

    def test_upstream_icdar_evaluator_correct_and_missing_predictions_blocked(self):
        gt=self.root/'gt.txt';gt.write_text('0,0,10,0,10,10,0,10,text\n')
        p=self.root/'pred.jsonl'
        write_jsonl(p,[dict(sample_id='img_1',polygons=[QUAD],scores=[.9])])
        result,_=icdar.evaluate(p,dict(img_1=gt))
        self.assertEqual(result['measured']['hmean'],1.)
        self.assertEqual(result['exact_paddle_83_8_reproduction'],'NOT VERIFIED')
        with self.assertRaisesRegex(ValueError,'incomplete'):
            icdar.evaluate(p,dict(img_1=gt,img_2=gt))

    def test_paddle_output_schema_and_scores(self):
        polys,scores=inference.parse_output([[0,0,10,0,10,10,0,10]],[.8])
        self.assertEqual(polys,[QUAD]);self.assertEqual(scores,[.8])
        with self.assertRaises(ValueError):inference.parse_output([],[.8])
        with self.assertRaises(ValueError):inference.parse_output([QUAD],[float('nan')])
        with self.assertRaises(ValueError):inference.parse_output([QUAD[:6]],[.8])
        with self.assertRaises(ValueError):inference.parse_output([QUAD],[1.5])

    def test_only_final_fragment_repaired_and_saved(self):
        p=self.root/'pred.jsonl';p.write_bytes(b'{"sample_id":"ok"}\n{"sample_id":')
        self.assertEqual(inference.resume(p),[dict(sample_id='ok')])
        self.assertTrue(p.with_suffix('.interrupted_tail').exists())
        p.write_bytes(b'{bad\n{"sample_id":"ok"}\n')
        with self.assertRaisesRegex(ValueError,'middle'):inference.resume(p)

    def test_task_builder_deduplicates_clean_and_retains_all_variants(self):
        p=self.root/'manifest.jsonl'
        row=dict(source_image_id='SOURCE',clean_relpath='clean/image.jpg',clean_sha256='a'*64,
                 variant_id='v1',degraded_relpath='main/v1.jpg',degraded_sha256='b'*64)
        write_jsonl(p,[row,dict(row,variant_id='v2',degraded_relpath='main/v2.jpg')])
        tasks=manifest.build_tasks(p,dict(clean_root='/clean',degraded_root='/deg'))
        self.assertEqual([t['sample_id'] for t in tasks],['clean::SOURCE','deg::v1','deg::v2'])
        self.assertFalse(any('gt' in k or 'homography' in k for t in tasks for k in t))
        write_jsonl(p,[row,row])
        with self.assertRaisesRegex(ValueError,'Duplicate'):manifest.build_tasks(p,dict(clean_root='/c',degraded_root='/d'))

    def test_two_gpu_controller_merges_once_and_rejects_changed_topology(self):
        tasks=[dict(sample_id=f'img_{i}',path=f'/fake/{i}',image_sha256='a'*64) for i in range(7)]
        model=dict(test_only=True)
        class FakeProcess:
            def __init__(self,cmd,**kwargs):
                get=lambda k:cmd[cmd.index(k)+1]
                records=[dict(sample_id=t['sample_id'],model_identity_sha256=get('--identity'),
                              manifest_sha256=get('--manifest-hash'),polygons=[],scores=[])
                         for t in jsonl(get('--tasks'))]
                write_jsonl(Path(get('--out'))/'predictions.jsonl',records)
                self.returncode=0
            def poll(self):return 0
            def wait(self,**kwargs):return 0
        with patch.object(inference.subprocess,'Popen',FakeProcess):
            merged=inference.launch(tasks,self.root/'out','/fake/model',model,'manifest','0,1')
            self.assertEqual(len(list(jsonl(merged))),7)
            with self.assertRaisesRegex(RuntimeError,'identity changed'):
                inference.launch(tasks,self.root/'out','/fake/model',model,'manifest','0')

    def test_phase1_controller_to_local_review_with_explicit_fake_detector(self):
        """Real data parsing/evaluation/rendering, synthetic 500-image fixture, no GPU."""
        images=self.root/'images';gt=self.root/'gt'
        images.mkdir();gt.mkdir()
        for i in range(1,501):
            Image.new('RGB',(20,20),'white').save(images/f'img_{i}.jpg')
            (gt/f'gt_img_{i}.txt').write_text('0,0,10,0,10,10,0,10,UNIT_TEST\n')
        cid=gate.clone_id(self.root)
        model=dict(config_sha256=sha256(inference.CONFIG),
                   config_content_sha256=identity(read_json(inference.CONFIG)),test_only=True)
        def fake_launch(tasks,output,model_dir,model_id,manifest_hash,gpus):
            p=Path(output)/'predictions.jsonl'
            write_jsonl(p,[dict(sample_id=t['sample_id'],model='paddle',polygons=[QUAD],scores=[.9],
                               model_identity_sha256=identity(model_id),manifest_sha256=manifest_hash) for t in tasks])
            return p
        out=self.root/'end_to_end'
        args=types.SimpleNamespace(model_dir='/fake',images=images,gt=gt,out=out,gpus='0,1',
                                   dataset_source='SYNTHETIC UNIT TEST ONLY; not a real ICDAR run')
        with patch.object(inference,'model_identity',return_value=model), \
             patch.object(inference,'launch',side_effect=fake_launch), \
             patch.object(gate,'clone_id',return_value=cid), \
             patch.object(workflow.subprocess,'check_output',return_value='synthetic test packages\n'):
            with run_lock(out):workflow.phase1(args)
        self.assertNotIn('.controller.lock',read_json(out/'phase1_result.json')['artifacts'])
        report=gate.verify_result(out,self.root)
        self.assertEqual(report['images'],500)
        gate.approve(out,'CPU test','Synthetic fixture only: tests parsing, counts, overlays and gate.',self.root)
        gate.require_phase1(model,self.root)


class FrozenArtifactsTests(unittest.TestCase):
    def test_all_frozen_hashes_and_decompressed_reference(self):
        with test_temp() as tmp:
            path=manifest.unpack(Path(tmp)/'reference.jsonl')
            lock=manifest.verify_shared(path)
            sources=set();variants=set();count=0
            for row in jsonl(path):
                count+=1;sources.add(row['source_image_id']);variants.add(row['variant_id'])
                self.assertEqual({g['ann_id'] for g in row['gt_clean']['valid']},
                                 {g['ann_id'] for g in row['gt_degraded']['valid']})
            self.assertEqual((count,len(sources),len(variants)),(9898,9898,9898))
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError,'Frozen shared artifact changed'):manifest.unpack(path)

    def test_public_notebook_source_matches_shared_builder(self):
        notebook=read_json(ROOT/'shared/Shared_Manifest_TextVQA_TextOCR_Public.ipynb')
        sources=[''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code']
        builder=(ROOT/'shared/shared_manifest.py').read_text(encoding='utf-8')
        self.assertTrue(any(s.startswith(builder.rstrip()) for s in sources))

    def test_all_notebooks_have_no_outputs_and_compile(self):
        for path in ROOT.rglob('*.ipynb'):
            if '.git' in path.parts:continue
            notebook=read_json(path)
            for cell in notebook['cells']:
                if cell['cell_type']=='code':
                    self.assertFalse(cell.get('outputs'))
                    self.assertIsNone(cell.get('execution_count'))
                    compile(''.join(cell['source']),str(path),'exec')

    def test_comparison_refuses_incompatible_hash(self):
        lock=manifest.verify_shared()
        monkey=read_json(ROOT/'monkey/phase2/report/metrics.json')
        paddle=copy.deepcopy(monkey);paddle['model']='paddle'
        with test_temp() as tmp:
            root=Path(tmp);save_json(root/'p.json',paddle)
            table=workflow.compare(root/'p.json',ROOT/'monkey/phase2/report/metrics.json',root/'comparison.md')
            self.assertIn('paddle',table.read_text())
            paddle['manifest_sha256']='wrong';save_json(root/'p.json',paddle)
            with self.assertRaisesRegex(ValueError,'different shared artifacts'):
                workflow.compare(root/'p.json',ROOT/'monkey/phase2/report/metrics.json',root/'comparison.md')


if __name__=='__main__':
    unittest.main()
