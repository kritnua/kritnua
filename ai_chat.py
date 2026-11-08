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
    payload={'model':model,'messages':[{'role':'system','content':system},{'role':'user','content':user}],
             'temperature':0,'max_completion_tokens':2048}
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
        hints={400:'Check the selected model and request settings.',401:'Check GROQ_API_KEY in Render.',403:'Check model access and project permissions.',404:'Select an active model from Groq Console.',429:'Groq free-plan quota or rate limit reached. Check Groq Console limits.'}
        raise HTTPException(502,f'Groq HTTP {response.status_code}. '+hints.get(response.status_code,'Groq is temporarily unavailable. Please retry later.'))
    try:
        data=response.json()
        choice=data['choices'][0]
        answer=choice['message']['content']
        if not isinstance(answer,str) or not answer.strip() or choice.get('finish_reason')=='length':
            raise ValueError()
        return answer.strip()
    except (ValueError,KeyError,IndexError,TypeError):
        raise HTTPException(502,'Groq returned an empty or incomplete answer. The local policy answer remains available.') from None
