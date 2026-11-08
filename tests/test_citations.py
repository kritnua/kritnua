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


def test_claims_use_validated_source_field_not_pdf_page_numbers():
    data={'supported':True,'claims':[{'text':'ลาได้ 120 วัน [13]','source':'Ref 1'},{'text':'รับค่าจ้างตามเงื่อนไข','source':'Ref 2'}]}
    answer,refs=validate_answer(json.dumps(data,ensure_ascii=False),EVIDENCE)
    assert '[13]' not in answer
    assert '120 วัน [Ref 1]' in answer
    assert refs==['Ref 1','Ref 2']


def test_claim_with_unknown_source_is_never_relabelled():
    data={'supported':True,'claims':[{'text':'ลาได้ 120 วัน','source':'Ref 13'}]}
    with pytest.raises(CitationError):validate_answer(json.dumps(data),EVIDENCE)


def test_claims_without_evidence_do_not_display_model_claim():
    data={'supported':False,'claims':[{'text':'999 days','source':'Ref 1'}]}
    answer,refs=validate_answer(json.dumps(data),EVIDENCE)
    assert '999' not in answer and not refs
