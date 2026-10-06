import copy
import csv
import json

import numpy as np
import pytest
import tifffile
import torch

from sic_xrt_ml.training.human_feedback import (
    CLASSES,
    digest,
    load_feedback,
    masked_loss,
    run,
    save,
    split_rows,
)


def fixtures(tmp_path, count=20):
    """Synthetic contract fixture, never delivered as real human annotations."""
    root=tmp_path/'feedback';root.mkdir();(root/'patches').mkdir();(root/'reviews').mkdir()
    samples,records,sources=[],[],[]
    for wafer in ('1','2'):
        rng=np.random.default_rng(int(wafer))
        source={'file_path':'synthetic_fixture_'+wafer+'.tif','sha256':wafer*64,'page_index':0,'width':512,'height':512,
                'coordinate_space':'raw_pixel_xy','orientation':'encoded_no_exif_rotation'}
        sid=source['sha256']+':0'
        events=[]
        for j in range(count):
            positive=j<count//2
            ident=f'w{wafer}_{j}';x,y=80.+(j%10)*20,180. if positive else 330.
            original={'candidate_id':ident,'x':x,'y':y,'predicted_type':'TSD','analysis_id':'fixture',
                      'model_sha256':'b'*64,'model_id':'synthetic'}
            prior={'x':x,'y':y,'objectness':None,'label':None,'location_confirmed':False,'duplicate_of':None}
            # Confirmed type and position are different events.
            event={'id':ident+'_1','record_id':ident,'revision':1,'action':'type' if positive else 'background',
                'reviewer':'synthetic test reviewer','actor':'human','created_at':'2026-10-06T00:00:00+00:00','note':'synthetic fixture',
                'original':original,'previous_target':prior,
                'target':{**prior,'objectness':positive,'label':CLASSES[j%3] if positive else None}}
            events.append(event)
            if positive:
                event={**event,'id':ident+'_2','revision':2,'action':'location','previous_target':copy.deepcopy(event['target']),
                       'target':{**event['target'],'location_confirmed':True}}
                events.append(event)
            record_key='session'+wafer+':'+ident
            record={'id':record_key,'source_id':sid,'wafer':wafer,'event':event}
            records.append(record)
            relative='patches/'+ident+'.tif'
            tifffile.imwrite(root/relative,rng.integers(0,256,(128,128,3),dtype=np.uint8),photometric='rgb')
            samples.append({'id':ident,'record_id':record_key,'source_id':sid,'wafer':wafer,'path':relative,'sha256':digest(root/relative),
                'role':'current','crop_xy':[round(x)-64,round(y)-64],'anchor_xy':[x,y],'objectness':int(positive),'objectness_mask':True,
                'type':event['target']['label'],'type_mask':positive,'offset_xy':[0.,0.],'location_mask':positive,
                'label_basis':'human_review','expert_ground_truth':False})
        review={'schema':'xrt_feedback_v1','session_id':'session'+wafer,'wafer_id':wafer,'source':source,'events':events,
                'expert_ground_truth':False,'exhaustive_annotations':False}
        relative='reviews/w'+wafer+'.json';save(root/relative,review);sha=digest(root/relative)
        for r in records:
            if r['wafer']==wafer:
                r['review_sha256']=sha
        for r in samples:
            if r['wafer']==wafer:
                r['review_sha256']=sha
        sources.append({'id':sid,'wafer':wafer,'source':source,'review_path':relative,'review_sha256':sha})
    save(root/'samples.json',samples);save(root/'records.json',records)
    info={'schema':'xrt_feedback_dataset_v1','input_schema':'xrt_feedback_v1','patch_contract':'rgb128_uint8_raw_pixel_xy',
        'max_offset_px':32,'classes':list(CLASSES),'samples_path':'samples.json','records_path':'records.json',
        'samples_sha256':digest(root/'samples.json'),'records_sha256':digest(root/'records.json'),'sources':sources,
        'sample_count':len(samples),'masks':{k:sum(r[k] for r in samples) for k in ('objectness_mask','type_mask','location_mask')},
        'expert_ground_truth':False,'exhaustive_annotations':False,'test_evaluated':False,'automatic_model_promotion':False}
    save(root/'dataset.json',info)
    retained=tmp_path/'retained';retained.mkdir();(retained/'patches').mkdir();old=[];old_sources=[]
    for wafer in ('1','2'):
        old_sources.append({'source_id':'old-source'+wafer,'sha256':('c' if wafer=='1' else 'd')*64,'wafer':wafer,'locator':'synthetic'})
        for index in range(15):
            ident=f'old{wafer}_{index}';relative='patches/'+ident+'.tif'
            tifffile.imwrite(retained/relative,np.random.default_rng(int(wafer)*100+index).integers(0,256,(128,128,3),dtype=np.uint8),photometric='rgb')
            old.append({'patch_id':ident,'source_id':'old-source'+wafer,'wafer':wafer,'label':CLASSES[index%3],
                        'path':relative,'sha256':digest(retained/relative),'x':180.,'y':180.})
    with (retained/'samples.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(old[0]));writer.writeheader();writer.writerows(old)
    save(retained/'sources.json',old_sources)
    save(retained/'cohort.json',{'schema':'wafer_grouped_development_cohort','schema_version':1,'manifest_sha256':digest(retained/'samples.csv'),
                               'expert_ground_truth':False,'test_evaluated':False})
    save(retained/'output_hashes.json',{'sources.json':digest(retained/'sources.json')})
    return root,retained


def test_unknown_type_and_location_masks_have_zero_gradient():
    presence=torch.zeros(2,requires_grad=True);kind=torch.randn(2,3,requires_grad=True);location=torch.randn(2,2,requires_grad=True)
    truth=torch.tensor([[1.,0.,0.,0.,1.,0.,0.],[0.,0.,0.,0.,1.,0.,0.]])
    loss,components=masked_loss((presence,kind,location),truth)
    loss.backward()
    assert presence.grad.abs().sum()>0
    assert kind.grad is None or kind.grad.abs().sum()==0
    assert location.grad is None or location.grad.abs().sum()==0
    assert components['type']==0 and components['location']==0


def test_package_tampering_holdout_and_leakage_rejected(tmp_path):
    root,_=fixtures(tmp_path)
    rows,_,_=load_feedback(root)
    assert len(rows)==40
    with pytest.raises(ValueError,match='8'):
        split_rows(rows,'8')
    rows[20]['source_id']=rows[0]['source_id']
    with pytest.raises(ValueError,match='different wafers'):
        split_rows(rows,'2')
    info=json.loads((root/'dataset.json').read_text())
    samples=json.loads((root/'samples.json').read_text());samples[0]['type_mask']=False
    save(root/'samples.json',samples);info['samples_sha256']=digest(root/'samples.json');save(root/'dataset.json',info)
    with pytest.raises(ValueError,match='Mask'):
        load_feedback(root)


def test_cpu_training_preserves_reference_data_and_never_deploys(tmp_path):
    root,retained=fixtures(tmp_path)
    report=run(root,retained,'2',tmp_path/'training',epochs=1)
    assert report['trained'] is True and report['status']=='completed_research'
    assert report['readiness']['train_location_head'] is True
    assert report['human_validation']['presence']['count']==20
    assert report['retained_validation_reference_agreement']['presence']['count']==15
    assert report['deployment_ready'] is False and report['deployed'] is False and report['test_evaluated'] is False
    assert (tmp_path/'training/research_model.pt').exists()
    assert report['comparison_checks']['retained_reference_non_decrease'] is False


def test_insufficient_feedback_writes_reasons_without_training(tmp_path):
    root,retained=fixtures(tmp_path,count=2)
    report=run(root,retained,'2',tmp_path/'training',epochs=1)
    assert report['trained'] is False and report['status']=='needs_review_data'
    assert len(report['reasons'])==2
    assert not (tmp_path/'training/research_model.pt').exists()
