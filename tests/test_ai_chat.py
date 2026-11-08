import asyncio
import httpx
import pytest
from fastapi import HTTPException
import ai_chat


def test_groq_response_and_secret_handling(monkeypatch):
    monkeypatch.setenv('GROQ_API_KEY','test-secret-key')
    monkeypatch.setenv('GROQ_MODEL','openai/gpt-oss-20b')
    original=httpx.AsyncClient
    def success(request):
        assert request.url.host=='api.groq.com'
        assert request.headers['Authorization']=='Bearer test-secret-key'
        return httpx.Response(200,json={'choices':[{'message':{'content':'คำตอบ [Ref 1]'},'finish_reason':'stop'}]})
    monkeypatch.setattr(ai_chat.httpx,'AsyncClient',lambda **kwargs:original(transport=httpx.MockTransport(success),**kwargs))
    assert asyncio.run(ai_chat.groq_complete('use evidence','question'))=='คำตอบ [Ref 1]'


@pytest.mark.parametrize('status',[401,403,404,429,503])
def test_groq_error_is_actionable_without_raw_provider_data(monkeypatch,status):
    monkeypatch.setenv('GROQ_API_KEY','test-secret-key')
    original=httpx.AsyncClient
    def handler(request):return httpx.Response(status,json={'error':{'message':'sensitive private payload'}})
    monkeypatch.setattr(ai_chat.httpx,'AsyncClient',lambda **kwargs:original(transport=httpx.MockTransport(handler),**kwargs))
    with pytest.raises(HTTPException) as error:
        asyncio.run(ai_chat.groq_complete('system','question'))
    assert str(status) in error.value.detail
    assert 'sensitive' not in error.value.detail and 'test-secret' not in error.value.detail


def test_groq_timeout_and_incomplete_output(monkeypatch):
    monkeypatch.setenv('GROQ_API_KEY','test-secret-key')
    original=httpx.AsyncClient
    def timeout(request):raise httpx.ReadTimeout('secret')
    monkeypatch.setattr(ai_chat.httpx,'AsyncClient',lambda **kwargs:original(transport=httpx.MockTransport(timeout),**kwargs))
    with pytest.raises(HTTPException) as error:asyncio.run(ai_chat.groq_complete('system','question'))
    assert error.value.status_code==504
    def incomplete(request):return httpx.Response(200,json={'choices':[{'message':{'content':'partial answer'},'finish_reason':'length'}]})
    monkeypatch.setattr(ai_chat.httpx,'AsyncClient',lambda **kwargs:original(transport=httpx.MockTransport(incomplete),**kwargs))
    with pytest.raises(HTTPException) as error:asyncio.run(ai_chat.groq_complete('system','question'))
    assert error.value.status_code==502
