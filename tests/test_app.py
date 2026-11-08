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
    assert a['mode']=='offline'
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
