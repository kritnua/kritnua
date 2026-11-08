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
    assert len(r.json()['results'])==7
    assert all(x['passed'] for x in r.json()['results'])


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
    assert '30 seconds' in repeated['warning'] and len(calls)==2
    assert repeated['citations']


def test_natural_question_uses_full_context_and_selected_groq_provider(monkeypatch):
    import app as module
    monkeypatch.setenv('CHAT_PROVIDER','groq')
    monkeypatch.setenv('GROQ_API_KEY','test-secret')
    monkeypatch.setattr(module,'_summary_cache',{})
    monkeypatch.setattr(module,'_gemini_unavailable_until',0.0)
    doc=client.post('/api/documents',files={'file':('Partner policy.txt','คู่สมรสคลอดบุตร\nลาได้ 12 วัน พร้อมค่าจ้าง\nใช้ภายใน 80 วันนับแต่คลอด'.encode())}).json()
    async def answer_from_context(system,user):
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
