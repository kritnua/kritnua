import os
import tempfile
os.environ['DATA_DIR']=tempfile.mkdtemp(prefix='hr-policy-test-')
os.environ.pop('GEMINI_API_KEY',None)
from fastapi.testclient import TestClient
from app import app

client=TestClient(app)

def test_full_document_lifecycle():
    assert client.get('/').status_code==200
    assert client.post('/api/ask',json={'question':'annual leave'}).json()['citations']==[]
    assert client.post('/api/documents',files={'file':('bad.exe',b'no')}).status_code==400
    r=client.post('/api/documents',files={'file':('leave.txt',b'Annual leave entitlement is 15 days per year. Please request annual leave 7 days in advance.')})
    assert r.status_code==200
    docid=r.json()['id']
    assert client.get('/api/documents').json()[0]['name']=='leave.txt'
    a=client.post('/api/ask',json={'question':'How many days of annual leave?'}).json()
    assert a['mode']=='policy'
    assert '15' in a['answer']
    assert a['citations'][0]['filename']=='leave.txt'
    assert a['citations'][0]['page']==1
    assert a['citations'][0]['reference'] in a['answer']
    assert a['citations'][0]['id'] not in a['answer']
    client.delete('/api/documents/'+docid)
    assert client.get('/api/documents').json()==[]

def test_sample_evaluation():
    assert client.post('/api/demo').status_code==200
    r=client.post('/api/evaluate')
    assert r.status_code==200
    assert r.json()['results']==[]
    assert r.json()['documents'][0]['status']=='NOT_CONFIGURED'
    assert r.json()['summary']['failed']==0


def test_provider_error_is_actionable_and_redacted():
    import httpx
    from app import provider_error
    for code, message, expected in [
        (429, 'quota contains sensitive value', 'โควตา'),
        (404, 'sensitive resource name', 'ไม่พบโมเดล'),
        (400, 'API key not valid sensitive', 'API key'),
        (403, 'private project sensitive', 'ไม่มีสิทธิ์'),
        (503, 'internal sensitive', 'ขัดข้อง'),
    ]:
        response = httpx.Response(code, json={'error': {'message': message}})
        error = provider_error(response, 'generation', 'test-model')
        assert error.status_code == 502
        assert expected in error.detail
        assert str(code) in error.detail
        assert 'sensitive' not in error.detail


def test_generation_text_handles_missing_and_multiple_parts():
    import pytest
    from fastapi import HTTPException
    from app import generation_text
    for data in ({}, {'candidates': []}, {'candidates': [{'content': None}]}, {'candidates': [{'content': {'parts':[{'thought':True,'text':'internal'}]}}]}):
        with pytest.raises(HTTPException) as exc:
            generation_text(data)
        assert exc.value.status_code == 502
    assert generation_text({'candidates':[{'content':{'parts':[{'thought':True,'text':'internal'},{'text':'Answer '},{'text':'[source]'}]}}]}) == 'Answer [source]'


def test_gemini_transport_errors(monkeypatch):
    import asyncio
    import httpx
    import pytest
    import app as module
    from fastapi import HTTPException
    original = httpx.AsyncClient
    for kind, expected in [('timeout',504),('connection',502),('invalid_json',502)]:
        def handler(request):
            if kind == 'timeout': raise httpx.ReadTimeout('secret must not appear')
            if kind == 'connection': raise httpx.ConnectError('secret must not appear')
            return httpx.Response(200, text='not json')
        monkeypatch.setattr(module.httpx, 'AsyncClient', lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
        with pytest.raises(HTTPException) as exc:
            asyncio.run(module.gemini_request('test','generateContent',{},'generation',1))
        assert exc.value.status_code == expected
        assert 'secret' not in exc.value.detail


def test_policy_search_and_gemini_fallback_without_embedding(monkeypatch):
    import app as module
    from fastapi import HTTPException
    monkeypatch.setattr(module, 'KEY', 'configured-for-test')
    async def no_embedding(*args, **kwargs):
        raise AssertionError('Search must not call embedding API')
    async def unavailable(*args, **kwargs):
        raise HTTPException(502, 'Gemini HTTP 503')
    monkeypatch.setattr(module, 'embed', no_embedding)
    monkeypatch.setattr(module, 'gemini_request', unavailable)
    direct=client.post('/api/ask',json={'question':'วันลาพักร้อนกี่วัน'}).json()
    assert direct['mode']=='policy'
    assert direct['citations'] and '15' in direct['answer']
    assert direct['generation_ms']==0 and 'warning' not in direct
    fallback=client.post('/api/ask',json={'question':'annual leave','use_gemini':True})
    assert fallback.status_code==200
    assert fallback.json()['mode']=='policy'
    assert '503' in fallback.json()['warning']
    assert '15' in fallback.json()['answer']
    async def success(model, operation, payload, stage, timeout):
        assert timeout == 25
        citation='Ref 1'
        return {'candidates':[{'content':{'parts':[{'text':f'15 days [{citation}]'}]}}]}
    monkeypatch.setattr(module, 'gemini_request', success)
    monkeypatch.setattr(module, '_gemini_unavailable_until', 0.0)
    summarized=client.post('/api/ask',json={'question':'annual leave','use_gemini':True}).json()
    assert summarized['mode']=='gemini'
    assert summarized['answer'].startswith('15 days')
    assert 'warning' not in summarized
    assert client.post('/api/ask',json={'question':'zzzxqvv'}).json()['citations']==[]


def test_upload_with_key_makes_no_provider_calls(monkeypatch):
    import app as module
    monkeypatch.setattr(module, 'KEY', 'configured')
    async def forbidden(*args, **kwargs):
        raise AssertionError('Upload must not call Google')
    monkeypatch.setattr(module, 'gemini_request', forbidden)
    response=client.post('/api/documents',files={'file':('independent.txt','ลาพักร้อน 18 วัน'.encode())})
    assert response.status_code==200
    assert response.json()['chunks']==1
    answer=client.post('/api/ask',json={'question':'ลาพักร้อน'}).json()
    assert '18' in answer['answer']
    client.delete('/api/documents/'+response.json()['id'])


def test_migration_preserves_ids_and_does_not_restore_deleted_documents():
    import chromadb
    from app import migrate_legacy_collections
    store=chromadb.PersistentClient(path=tempfile.mkdtemp(prefix='hr-migrate-'))
    old=store.create_collection('policies-legacy')
    old.add(ids=['original-1'],documents=['Annual leave 18 days'],metadatas=[{'document_id':'original','filename':'legacy.txt','page':2}],embeddings=[[0.1,0.2,0.3]])
    target=store.create_collection('policy-local-v1')
    migrate_legacy_collections(store,target)
    assert target.count()==1
    assert target.get()['ids']==['original-1']
    assert target.get()['metadatas'][0]['page']==2
    target.delete(ids=['original-1'])
    migrate_legacy_collections(store,store.get_collection('policy-local-v1'))
    assert target.count()==0
    assert old.count()==1


def test_summary_cache_and_failure_cooldown(monkeypatch):
    import app as module
    from fastapi import HTTPException
    monkeypatch.setattr(module,'KEY','test-configured')
    monkeypatch.setattr(module,'_summary_cache',{})
    monkeypatch.setattr(module,'_gemini_unavailable_until',0.0)
    calls=[]
    async def success(*args,**kwargs):
        calls.append(1)
        return {'candidates':[{'content':{'parts':[{'text':'15 days [Ref 1]'}]}}]}
    monkeypatch.setattr(module,'gemini_request',success)
    first=client.post('/api/ask',json={'question':'annual leave','use_gemini':True}).json()
    second=client.post('/api/ask',json={'question':'annual leave','use_gemini':True}).json()
    assert first['mode']==second['mode']=='gemini'
    assert second['cached'] is True and len(calls)==1
    async def unavailable(*args,**kwargs):
        calls.append(1)
        raise HTTPException(502,'HTTP 503')
    monkeypatch.setattr(module,'gemini_request',unavailable)
    fallback=client.post('/api/ask',json={'question':'probation period','use_gemini':True}).json()
    assert fallback['mode']=='policy' and '503' in fallback['warning']
    repeated=client.post('/api/ask',json={'question':'probation period','use_gemini':True}).json()
    assert 'seconds' in repeated['warning'] and len(calls)==2
    assert '503' in repeated['warning']
    assert client.get('/api/health').json()['last_ai_error']=='HTTP 503'
    assert repeated['citations']


def test_natural_question_uses_full_context_and_selected_groq_provider(monkeypatch):
    import app as module
    monkeypatch.setenv('CHAT_PROVIDER','groq')
    monkeypatch.setenv('GROQ_API_KEY','test-secret')
    monkeypatch.setattr(module,'_summary_cache',{})
    monkeypatch.setattr(module,'_gemini_unavailable_until',0.0)
    doc=client.post('/api/documents',files={'file':('Partner policy.txt','คู่สมรสคลอดบุตร\nลาได้ 12 วัน พร้อมค่าจ้าง\nใช้ภายใน 80 วันนับแต่คลอด'.encode())}).json()
    async def answer_from_context(system,user,allowed_references):
        assert 'Ref 1' in allowed_references
        assert 'อยากอยู่ช่วยแฟนหลังมีน้อง' in user
        assert 'ลาได้ 12 วัน' in user
        assert 'Sample HR Policy' not in user
        return 'ลาได้ 12 วัน พร้อมค่าจ้าง ใช้ภายใน 80 วันนับแต่คลอด [Ref 1]'
    monkeypatch.setattr(module,'groq_complete',answer_from_context)
    result=client.post('/api/ask',json={'question':'อยากอยู่ช่วยแฟนหลังมีน้อง','use_ai':True,'document_id':doc['id']}).json()
    assert result['mode']=='groq' and result['provider_label']=='Groq'
    assert result['context_complete'] and '12 วัน' in result['answer']
    assert all(c['filename']=='Partner policy.txt' for c in result['citations'])
    assert client.get('/api/health').json()['provider_label']=='Groq'
    client.delete('/api/documents/'+doc['id'])


def test_structured_groq_answer_and_citation_failures_do_not_mark_provider_down(monkeypatch):
    import json
    import app as module
    monkeypatch.setenv('CHAT_PROVIDER','groq')
    monkeypatch.setenv('GROQ_API_KEY','test-key')
    monkeypatch.setattr(module,'_summary_cache',{})
    monkeypatch.setattr(module,'_gemini_unavailable_until',0.0)
    async def invalid(*args):return 'Answer without source references'
    monkeypatch.setattr(module,'groq_complete',invalid)
    failed=client.post('/api/ask',json={'question':'annual leave','use_ai':True}).json()
    assert failed['mode']=='policy'
    assert 'retry AI now' in failed['warning']
    assert module._gemini_unavailable_until==0.0
    async def valid(*args):return json.dumps({'answer':'15 days','supported':True,'references':['Ref 1']})
    monkeypatch.setattr(module,'groq_complete',valid)
    passed=client.post('/api/ask',json={'question':'annual leave','use_ai':True}).json()
    assert passed['mode']=='groq'
    assert '[Ref 1]' in passed['answer']
    assert client.get('/api/health').json()['last_ai_error'] is None
    assert client.head('/').status_code==200


def test_real_groq_adapter_uses_source_enum_on_page_thirteen(monkeypatch):
    import asyncio,json,httpx
    import app as module
    import ai_chat
    monkeypatch.setenv('CHAT_PROVIDER','groq')
    monkeypatch.setenv('GROQ_API_KEY','test-key')
    monkeypatch.setenv('GROQ_MODEL','openai/gpt-oss-20b')
    monkeypatch.setattr(module,'_summary_cache',{})
    monkeypatch.setattr(module,'_gemini_unavailable_until',0.0)
    doc=asyncio.run(module.ingest('Synthetic family policy.pdf',['']*12+['ลาคลอดบุตร\nลาได้ไม่เกิน 100 วัน\nได้รับค่าจ้างไม่เกิน 50 วัน']))
    original=httpx.AsyncClient
    def provider(request):
        payload=json.loads(request.content)
        allowed=payload['response_format']['json_schema']['schema']['properties']['claims']['items']['properties']['source']['enum']
        assert allowed==['Ref 1']
        assert 'PAGE: 13' in payload['messages'][1]['content']
        output=json.dumps({'supported':True,'claims':[{'text':'ลาได้ไม่เกิน 100 วัน','source':'Ref 1'},{'text':'ได้รับค่าจ้างไม่เกิน 50 วัน','source':'Ref 1'}]},ensure_ascii=False)
        return httpx.Response(200,json={'choices':[{'message':{'content':output},'finish_reason':'stop'}]})
    monkeypatch.setattr(ai_chat.httpx,'AsyncClient',lambda **kwargs:original(transport=httpx.MockTransport(provider),**kwargs))
    response=client.post('/api/ask',json={'question':'ลาคลอด','use_ai':True,'document_id':doc['id']}).json()
    assert response['mode']=='groq' and 'warning' not in response
    assert '100 วัน [Ref 1]' in response['answer']
    assert response['citations'][0]['page']==13
    assert client.get('/api/health').json()['last_ai_error'] is None
    client.delete('/api/documents/'+doc['id'])


def test_english_question_gets_english_answer_with_original_thai_evidence(monkeypatch):
    import json
    import app as module
    monkeypatch.setenv('CHAT_PROVIDER','groq')
    monkeypatch.setenv('GROQ_API_KEY','test-key')
    monkeypatch.setattr(module,'_summary_cache',{})
    monkeypatch.setattr(module,'_gemini_unavailable_until',0.0)
    doc=client.post('/api/documents',files={'file':('Thai leave policy.txt','ลาป่วย\nลาได้ตามที่ป่วยจริง\nได้รับค่าจ้างไม่เกิน 24 วันทำงานต่อปี'.encode())}).json()
    async def respond(system,user,allowed):
        assert 'Write every answer point in English' in system
        assert '24 วันทำงานต่อปี' in user
        return json.dumps({'supported':True,'claims':[{'text':'You may take sick leave when actually ill. Paid sick leave is limited to 24 working days per year.','source':allowed[0]}]})
    monkeypatch.setattr(module,'groq_complete',respond)
    result=client.post('/api/ask',json={'question':'If I am ill, can I take leave and will I be paid?','use_ai':True,'document_id':doc['id']}).json()
    assert result['answer_language']=='en' and result['mode']=='groq'
    assert 'Paid sick leave is limited to 24 working days' in result['answer']
    assert 'ลาป่วย' in result['citations'][0]['text']
    client.delete('/api/documents/'+doc['id'])


def test_long_thai_manual_uses_english_query_translation_before_context_search(monkeypatch):
    import asyncio,json
    import app as module
    monkeypatch.setenv('CHAT_PROVIDER','groq')
    monkeypatch.setenv('GROQ_API_KEY','test-key')
    monkeypatch.setattr(module,'_summary_cache',{})
    monkeypatch.setattr(module,'_search_query_cache',{})
    monkeypatch.setattr(module,'_gemini_unavailable_until',0.0)
    pages=['กฎการแต่งกายและเครื่องแบบพนักงาน\n'*80 for _ in range(8)]+['ลาป่วย\nลาได้ตามที่ป่วยจริง\nได้รับค่าจ้างไม่เกิน 24 วันทำงานต่อปี']
    doc=asyncio.run(module.ingest('Long Thai policy.pdf',pages))
    translations=[]
    async def translate(question):translations.append(question);return 'สิทธิลาป่วยและการได้รับค่าจ้าง'
    async def respond(system,user,allowed):
        assert 'PAGE: 9' in user and '24 วันทำงานต่อปี' in user
        assert 'Write every answer point in English' in system
        return json.dumps({'supported':True,'claims':[{'text':'Paid sick leave is limited to 24 working days per year.','source':allowed[0]}]})
    monkeypatch.setattr(module,'groq_search_query',translate)
    monkeypatch.setattr(module,'groq_complete',respond)
    payload={'question':'If I become ill, will I still receive my salary?','use_ai':True,'document_id':doc['id']}
    result=client.post('/api/ask',json=payload).json()
    assert result['mode']=='groq' and result['translated_search']
    assert result['answer_language']=='en' and result['citations'][0]['page']==9
    again=client.post('/api/ask',json=payload).json()
    assert again['cached'] and len(translations)==1
    client.delete('/api/documents/'+doc['id'])
