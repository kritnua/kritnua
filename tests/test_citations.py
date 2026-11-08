import json
import pytest
from citations import validate_answer,CitationError

EVIDENCE=[{'reference':'Ref 1'},{'reference':'Ref 2'}]

@pytest.mark.parametrize('raw',[
    'ลาป่วยตามจริง [Ref 1, Ref 2]',
    'ลาป่วยตามจริง [1, 2]',
    'ลาป่วยตามจริง [ref1; 2]',
    'ลาป่วยตามจริง [Ref1] [Ref 2]',
])
def test_common_inline_styles_are_normalized(raw):
    answer,refs=validate_answer(raw,EVIDENCE)
    assert refs==['Ref 1','Ref 2']
    assert '[Ref 1]' in answer and '[Ref 2]' in answer


def test_structured_answer_uses_separate_references():
    raw=json.dumps({'answer':'ลาป่วยได้ตามที่ป่วยจริง','supported':True,'references':['Ref 1',2]},ensure_ascii=False)
    answer,refs=validate_answer(raw,EVIDENCE)
    assert answer.startswith('ลาป่วยได้ตามที่ป่วยจริง')
    assert refs==['Ref 1','Ref 2']
    assert '[Ref 1]' in answer and '[Ref 2]' in answer


@pytest.mark.parametrize('raw',[
    'คำตอบ [Ref 99]',
    'คำตอบ [1, 99]',
    'คำตอบที่ไม่มีการอ้างอิง',
    '{"answer":"unsupported source","supported":true,"references":["Ref 99"]}',
    '{"answer":"no reference array","supported":true}',
])
def test_unknown_or_missing_sources_are_rejected(raw):
    with pytest.raises(CitationError):validate_answer(raw,EVIDENCE)


def test_model_admitting_insufficient_evidence_does_not_show_unsupported_claim():
    answer,refs=validate_answer(json.dumps({'answer':'ลาได้ 999 วัน','supported':False,'references':[]},ensure_ascii=False),EVIDENCE)
    assert '999' not in answer and refs==[]
    assert 'ไม่พบข้อมูลเพียงพอ' in answer
