import asyncio
from copy import deepcopy
import policy_evaluation as evaluation


def response_for(case,docid='d',name='Manual_Dugong.pdf'):
    text='\n'.join(expected for label,expected in case['clauses'])
    return {'answer':text+' [Ref 1]','citations':[{'id':'piece','reference':'Ref 1','document_id':docid,'filename':name,'page':case['page'],'text':text}]}


def test_sick_leave_checks_conditions_not_just_number_30():
    case=next(c for c in evaluation.CASES if c['id']=='sick_leave')
    doc={'id':'d','name':'Manual_Dugong.pdf'}
    correct=response_for(case)
    assert evaluation.assess(case,correct,doc)['status']=='PASS'
    wrong=deepcopy(correct);wrong['answer']='ลาป่วยได้ 30 วัน [Ref 1]'
    assert evaluation.assess(case,wrong,doc)['status']=='FAIL'
    wrong=deepcopy(correct);wrong['citations'][0]['filename']='Another employer.pdf'
    assert evaluation.assess(case,wrong,doc)['status']=='FAIL'
    wrong=deepcopy(correct);wrong['citations'][0]['page']=9
    assert evaluation.assess(case,wrong,doc)['status']=='FAIL'
    wrong=deepcopy(correct);wrong['answer']=wrong['answer'].replace('[Ref 1]','')
    assert evaluation.assess(case,wrong,doc)['status']=='FAIL'


def test_unknown_document_and_changed_version_are_not_test_failures():
    rows={'ids':['x-1','y-1'],'documents':['Unreviewed policy','Changed source'],
          'metadatas':[{'document_id':'x','filename':'New policy.pdf','page':1},{'document_id':'y','filename':'Manual_Dugong.pdf','page':1}]}
    async def must_not_run(*args):raise AssertionError('Unverified documents must not be graded')
    result=asyncio.run(evaluation.evaluate(rows,must_not_run))
    assert result['summary']=={'passed':0,'failed':0,'total':0,'unverified_documents':2}
    assert {d['status'] for d in result['documents']}=={'NOT_CONFIGURED','VERSION_MISMATCH'}


def test_reviewed_profile_runs_every_case_scoped_to_document(monkeypatch):
    rows={'ids':['d-1'],'documents':['Reviewed source'], 'metadatas':[{'document_id':'d','filename':'Reviewed.pdf','page':1}]}
    from policy_qa import reconstruct_pages
    monkeypatch.setattr(evaluation,'PROFILES',{'reviewed':{'filename':'Reviewed.pdf','fingerprint':evaluation.fingerprint(reconstruct_pages(rows))}})
    calls=[]
    async def respond(question,docid):
        calls.append((question,docid))
        case=next(c for c in evaluation.CASES if c['question']==question)
        return response_for(case,docid,'Reviewed.pdf')
    result=asyncio.run(evaluation.evaluate(rows,respond,'d'))
    assert result['summary']['passed']==8 and result['summary']['failed']==0
    assert all(docid=='d' for question,docid in calls)
    assert result['scope']=='local_policy_answers'
