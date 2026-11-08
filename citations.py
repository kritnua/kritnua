"""Normalize model citation syntax without inventing source associations."""
import json
import re
from fastapi import HTTPException

class CitationError(HTTPException):
    def __init__(self, detail):
        super().__init__(502, detail)


def normalize_reference(value):
    if isinstance(value,int) and not isinstance(value,bool):return f'Ref {value}'
    if not isinstance(value,str):raise CitationError('AI returned an invalid source reference.')
    match=re.fullmatch(r'\s*\[?\s*(?:ref(?:erence)?\s*)?(\d+)\s*\]?\s*',value,re.I)
    if not match:raise CitationError('AI returned an invalid source reference.')
    return f'Ref {int(match.group(1))}'


def validate_answer(raw,citations):
    """Accept structured answers and common reference styles; reject unknown IDs."""
    allowed={c['reference'] for c in citations}
    references=[]
    supported=True
    structured=False
    if not isinstance(raw,str):raise CitationError('AI returned an invalid answer format.')
    answer=raw.strip()
    if answer.startswith('```'):
        answer=re.sub(r'^```(?:json)?\s*|\s*```$','',answer,flags=re.I).strip()
    if answer.startswith('{'):
        try:data=json.loads(answer)
        except ValueError:raise CitationError('AI returned malformed structured answer data.') from None
        if isinstance(data,dict) and 'claims' in data:
            if not isinstance(data.get('supported'),bool) or not isinstance(data['claims'],list):
                raise CitationError('AI returned invalid claim data.')
            if not data['supported']:
                return 'ไม่พบข้อมูลเพียงพอในหลักฐานที่ส่งให้ AI จึงยังตอบคำถามนี้ไม่ได้ กรุณาระบุคู่มือหรือรายละเอียดเพิ่มเติม',[]
            lines=[];used=[]
            for claim in data['claims']:
                if not isinstance(claim,dict) or not isinstance(claim.get('text'),str) or not claim['text'].strip():
                    raise CitationError('AI returned an empty answer point.')
                ref=normalize_reference(claim.get('source'))
                if ref not in allowed:
                    raise CitationError('AI cited a source outside the supplied policy evidence. The local answer is shown instead.')
                # Only the validated source field creates clickable references.
                text=re.sub(r'\[(?:ref(?:erence)?\s*)?\d+(?:\s*[,;]\s*(?:ref(?:erence)?\s*)?\d+)*\]','',claim['text'],flags=re.I).strip()
                if not text:raise CitationError('AI returned an empty answer point.')
                lines.append(f'{text} [{ref}]');used.append(ref)
            if not lines:raise CitationError('AI returned no supported answer points.')
            return '\n\n'.join(lines),list(dict.fromkeys(used))
        if not isinstance(data,dict) or not isinstance(data.get('answer'),str) or not data['answer'].strip():
            raise CitationError('AI returned an empty structured answer.')
        if not isinstance(data.get('supported'),bool) or not isinstance(data.get('references'),list):
            raise CitationError('AI returned incomplete source reference data.')
        structured=True
        answer=data['answer'].strip()
        supported=data['supported']
        references=[normalize_reference(r) for r in data['references']]
    if any(r not in allowed for r in references):
        raise CitationError('AI cited a source outside the supplied policy evidence. The local answer is shown instead.')
    if structured and not supported:
        # Do not display an unsupported model claim as an evidence-backed answer.
        return 'ไม่พบข้อมูลเพียงพอในหลักฐานที่ส่งให้ AI จึงยังตอบคำถามนี้ไม่ได้ กรุณาระบุคู่มือหรือรายละเอียดเพิ่มเติม',[]

    def replace_group(match):
        group=match.group(1)
        # Normalize [1], [Ref1], [Ref 1, Ref 2], [1, 2], [ref 1; 2].
        items=re.split(r'\s*[,;]\s*',group)
        refs=[normalize_reference(item) for item in items]
        if any(r not in allowed for r in refs):
            raise CitationError('AI cited a source outside the supplied policy evidence. The local answer is shown instead.')
        references.extend(refs)
        return ' '.join(f'[{r}]' for r in dict.fromkeys(refs))
    pattern=r'\[((?:(?:ref(?:erence)?\s*)?\d+)(?:\s*[,;]\s*(?:ref(?:erence)?\s*)?\d+)*)\]'
    answer=re.sub(pattern,replace_group,answer,flags=re.I)
    # An explicit malformed Ref label must not be silently ignored.
    for label in re.findall(r'\[([^\]]+)\]',answer):
        if re.match(r'^ref',label,re.I) and label not in allowed:
            raise CitationError('AI returned a source label that cannot be matched to the supplied evidence.')
    references=list(dict.fromkeys(references))
    if not references:
        raise CitationError('AI omitted source references. The local policy answer is shown instead.')
    present=set(re.findall(r'\[(Ref \d+)\]',answer))
    missing=[r for r in references if r not in present]
    if missing:answer+='\n\n'+' '.join(f'[{r}]' for r in missing)
    return answer,references
