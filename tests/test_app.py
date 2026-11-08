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
    assert a['citations'][0]['id'] in a['answer']
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
        assert timeout == 20
        citation=module.retrieve_local('annual leave')[0]['id']
        return {'candidates':[{'content':{'parts':[{'text':f'15 days [{citation}]'}]}}]}
    monkeypatch.setattr(module, 'gemini_request', success)
    summarized=client.post('/api/ask',json={'question':'annual leave','use_gemini':True}).json()
    assert summarized['mode']=='gemini'
    assert summarized['answer'].startswith('15 days')
    assert 'warning' not in summarized
    assert client.post('/api/ask',json={'question':'zzzxqvv'}).json()['citations']==[]
