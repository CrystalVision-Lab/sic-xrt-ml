import pytest

torch = pytest.importorskip('torch')

from sic_xrt_ml.training.center_background import (
    FocusedBackgroundCNN,
    fold_ids,
    validate_added,
)
from sic_xrt_ml.training.independent_background import IndependentBackgroundCNN


def test_views_resist_global_brightness_and_gain():
    torch.manual_seed(3)
    model = FocusedBackgroundCNN()
    image = torch.rand(2,3,128,128)*.3+.25
    original = model.views(image)
    changed = model.views(image*.8+.1)
    for a,b in zip(original,changed,strict=True):
        assert a.shape == (2,3,64,64)
        torch.testing.assert_close(a,b,atol=5e-5,rtol=5e-5)


def test_flat_input_finite_and_backward_and_same_parameter_count():
    model = FocusedBackgroundCNN()
    image = torch.full((2,3,128,128),.5,requires_grad=True)
    output = model(image)
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert torch.isfinite(image.grad).all()
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in IndependentBackgroundCNN().parameters())


def test_center_view_ignores_far_corner_object():
    torch.manual_seed(4)
    image = torch.rand(1,3,128,128)*.2+.3
    changed = image.clone()
    changed[:,:,:8,:8] = 1.
    model = FocusedBackgroundCNN()
    torch.testing.assert_close(model.views(image)[0],model.views(changed)[0])


def test_fold_excludes_wafer_and_uncertain_and_only_adds_training_negatives():
    rows = [{'wafer':w,'train_eligible':eligible,'additional':added}
            for w in ('1','2','9') for eligible,added in ((True,False),(True,True),(False,True))]
    base,held = fold_ids(rows,'2',False)
    extended,held2 = fold_ids(rows,'2',True)
    assert held == held2 == [3,4,5]
    assert base == [0,6]
    assert extended == [0,1,6,7]
    with pytest.raises(ValueError,match='Development'):
        fold_ids(rows+[{'wafer':'8'}],'2',True)


def fixture_added():
    info = {'cohort_manifest_sha256':'cohort','coordinate_repair_sha256':'repair'}
    rows = [{'id':w,'wafer':w,'sha256':'hash'+w,'review_actor':'Codex_AI','human_verified':False,
             'expert_ground_truth':False,'decision':'background_candidate','weak_training_eligible':True}
            for w in ('1','2','9')]
    return {'schema':'xrt_weak_background_v1','test_evaluated':False,**info,'candidates':rows},info


@pytest.mark.parametrize('kind',['id','hash','holdout','eligibility','human','manifest'])
def test_additional_contract_rejects_leakage_and_provenance_errors(kind):
    data,info = fixture_added()
    ids,hashes = [],[]
    if kind=='id':
        ids=['1']
    elif kind=='hash':
        hashes=['hash1']
    elif kind=='holdout':
        data['candidates'][0]['wafer']='8'
    elif kind=='eligibility':
        data['candidates'][0]['decision']='uncertain'
    elif kind=='human':
        data['candidates'][0]['human_verified']=True
    else:
        data['cohort_manifest_sha256']='wrong'
    with pytest.raises(ValueError):
        validate_added(data,info,ids,hashes)


def test_additional_valid_contract():
    data,info = fixture_added()
    assert len(validate_added(data,info,[],[])) == 3
