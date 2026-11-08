"""Optional Groq chat adapter. Document processing remains entirely local."""
import asyncio
import os
import httpx
from fastapi import HTTPException

async def groq_complete(system, user):
    key=os.getenv('GROQ_API_KEY','').strip()
    model=os.getenv('GROQ_MODEL','openai/gpt-oss-20b').strip()
    if not key:
        raise HTTPException(503,'Groq API key is not configured. The local policy answer remains available.')
    schema_instruction=' Return only a JSON object with keys: answer (string), supported (boolean), references (array of source labels such as \"Ref 1\"). Use only source labels present in evidence. If evidence is insufficient set supported=false and references=[].'
    payload={'model':model,'messages':[{'role':'system','content':system+schema_instruction},{'role':'user','content':user}],
             'response_format':{'type':'json_object'},
             'temperature':0,'max_completion_tokens':1024}
    if model.startswith('openai/gpt-oss-'):
        payload['reasoning_effort']='low'
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            response=await asyncio.wait_for(client.post('https://api.groq.com/openai/v1/chat/completions',headers={'Authorization':f'Bearer {key}'},json=payload),timeout=20)
    except (httpx.TimeoutException,asyncio.TimeoutError):
        raise HTTPException(504,'Groq timed out. The local policy answer remains available.') from None
    except httpx.RequestError:
        raise HTTPException(502,'Could not connect to Groq. The local policy answer remains available.') from None
    if response.status_code!=200:
        try:
            error=response.json().get('error',{})
            code=str(error.get('code','')).lower()
            message=str(error.get('message','')).lower()
        except (ValueError,AttributeError):
            code,message='',''
        # Inspect provider data for categories, but never return its raw text.
        hints={400:'Check GROQ_MODEL and request settings.',401:'GROQ_API_KEY is invalid or expired. Replace it in this Render service.',403:'The account has no permission to use this model. Check Groq model permissions.',404:'GROQ_MODEL was not found. Select an active model in Groq Console.',429:'Groq quota or rate limit reached. Check Usage and Limits in Groq Console.'}
        hint=hints.get(response.status_code,'Groq is temporarily unavailable. Please retry later.')
        if code in {'model_decommissioned','model_not_found'} or 'decommissioned' in message:
            hint='This model is unavailable or retired. Change GROQ_MODEL to an active model permitted for your account.'
        elif 'request too large' in message or 'context length' in message or code=='context_length_exceeded':
            hint='The policy context exceeds the model or account token limit. Select one policy document and retry.'
        elif response.status_code==429 and ('tokens per minute' in message or 'tpm' in message or code=='tokens'):
            hint='Groq tokens-per-minute limit reached. Wait for the quota reset and select one policy document.'
        elif code=='invalid_api_key':
            hint='GROQ_API_KEY is invalid. Replace it in this Render service.'
        retry=response.headers.get('retry-after','')
        if retry.isdigit():hint+=f' Retry after {min(int(retry),86400)} seconds.'
        raise HTTPException(502,f'Groq HTTP {response.status_code}. '+hint)
    try:
        data=response.json()
        choice=data['choices'][0]
        answer=choice['message']['content']
        if not isinstance(answer,str) or not answer.strip() or choice.get('finish_reason')=='length':
            raise ValueError()
        return answer.strip()
    except (ValueError,KeyError,IndexError,TypeError):
        raise HTTPException(502,'Groq returned an empty or incomplete answer. The local policy answer remains available.') from None
