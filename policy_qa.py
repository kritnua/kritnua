"""Local, extractive policy answers. Never invent entitlements or combine employers."""
import hashlib
import math
import re
import unicodedata
from collections import Counter, defaultdict

TOPICS = {
    'hours': ('เวลาปฏิบัติงาน','เวลาทำงาน','เวลางาน','เข้างาน','เลิกงาน','working hours','work hours','office hours','working time'),
    'sick': ('ลาป่วย','sick leave'),
    'annual': ('ลาพักร้อน','ลาพักผ่อน','วันหยุดพักผ่อน','annual leave','vacation'),
    'personal': ('ลากิจ','personal leave'),
    'maternity': ('ลาคลอด','maternity leave','parental leave'),
    'ordination': ('ลาอุปสมบท','ลาบวช','ordination leave'),
    'military': ('ลารับราชการทหาร','military leave'),
}

def compact(text):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', text).lower()).translate(str.maketrans('๐๑๒๓๔๕๖๗๘๙','0123456789'))

def topics_in(text):
    value=compact(text)
    return {topic for topic, aliases in TOPICS.items() if any(compact(a) in value for a in aliases)}

def terms(text):
    value=unicodedata.normalize('NFKC',text).lower()
    stop={'how','many','much','what','is','the','a','an','of','per','do','does','i','my','to','in','and','for','are','can','should'}
    words=[w for w in re.findall(r'[a-z0-9]+',value) if w not in stop]
    thai=''.join(re.findall(r'[\u0e00-\u0e7f]+',value))
    words.extend(thai[i:i+3] for i in range(max(0,len(thai)-2)))
    return words

def page_chunks(text, size=1100, overlap=200):
    """Keep heading and following lines together; split at line boundaries."""
    text=text.replace('\r\n','\n').strip()
    offset=0
    while offset < len(text):
        end=min(offset+size,len(text))
        if end<len(text):
            boundary=text.rfind('\n',offset+size//2,end)
            if boundary>offset: end=boundary
        piece=text[offset:end]
        if piece.strip(): yield piece,offset
        if end>=len(text): break
        offset=max(offset+1,end-overlap)


def reconstruct_pages(rows):
    grouped=defaultdict(list)
    for ident,text,meta in zip(rows['ids'],rows['documents'],rows['metadatas']):
        grouped[(meta['document_id'],meta['page'])].append((ident,text,meta))
    pages=[]
    for (docid,page),items in grouped.items():
        def order(item):
            m=item[2]
            if 'offset' in m: return m['offset']
            try: return int(item[0].rsplit('-',1)[1])
            except (IndexError,ValueError): return 0
        items.sort(key=order)
        combined='';last_end=0
        for ident,text,meta in items:
            if meta.get('index_version')==2:
                offset=meta['offset']
                original_length=len(text)
                if combined and offset<last_end:
                    text=text[max(0,last_end-offset):]
                    combined+=text
                else: combined+= ('\n' if combined else '')+text
                last_end=max(last_end,offset+original_length)
            else:
                # Old indexes stored each line; rejoin before finding an answer.
                overlap=0
                for n in range(min(200,len(combined),len(text)),29,-1):
                    if combined.endswith(text[:n]): overlap=n;break
                combined+=text[overlap:] if overlap else ('\n' if combined else '')+text
        pages.append({'document_id':docid,'page':page,'filename':items[0][2]['filename'],'text':combined})
    return pages


def is_toc(text):
    header=compact(text[:250])
    return 'สารบัญ' in header or 'tableofcontents' in header or len(re.findall(r'\.{3,}\s*\d+',text))>=3


def relevant_units(page, wanted):
    """Extract an anchor with following detail, stopping at another policy topic."""
    text=page['text']
    # Split English sentences and Thai bullet paragraphs without splitting times.
    lines=[line.strip() for line in re.split(r'\n+|(?<=[.!?])\s+(?=[A-Z])',text) if line.strip()]
    if wanted:
        for i,line in enumerate(lines):
            if not (topics_in(line)&wanted): continue
            section=[line]
            for following in lines[i+1:i+10]:
                other=topics_in(following)
                if re.match(r'^[A-Z][A-Za-z -]{2,35}:',following) and not other&wanted: break
                if other and not other&wanted: break
                # A numbered heading for an unrelated section ends the excerpt.
                if sum(map(len,section))+len(following)>1500: break
                section.append(following)
            yield '\n'.join(section)
    else:
        for piece,offset in page_chunks(text): yield piece


def has_fact(text,wanted):
    value=compact(text)
    if wanted=={'hours'}:
        return bool(re.search(r'\d{1,2}[:.]\d{2}|\d+(?:โมง|นาฬิกา|ชั่วโมง|hours?)',value))
    if wanted:
        return bool(re.search(r'\d+(?:\.\d+)?(?:วัน|เดือน|สัปดาห์|days?|months?|weeks?)',value) or
                    re.search(r'(?:หนึ่ง|สอง|สาม|ห้า|หก|เจ็ด|สิบ|ยี่สิบ|สามสิบ|เก้าสิบ)(?:วัน|เดือน)',value) or
                    'ตามที่ป่วยจริง' in value)
    return len(text)>45


def retrieve(question,rows,document_id=None):
    wanted=topics_in(question)
    query=set(terms(question))
    if not query:return []
    candidates=[]
    for page in reconstruct_pages(rows):
        if document_id and page['document_id']!=document_id:continue
        if is_toc(page['text']):continue
        for text in relevant_units(page,wanted):
            tokens=Counter(terms(text));overlap=query.intersection(tokens)
            # Exact topic synonyms support English questions about Thai policies.
            topic_match=bool(wanted and topics_in(text)&wanted)
            if not topic_match and (not overlap or len(overlap)/len(query)<0.2):continue
            fact=has_fact(text,wanted)
            if wanted and not fact:continue
            score=sum(1+math.log1p(tokens[t]) for t in overlap)/math.sqrt(max(1,len(tokens))/30)
            if topic_match:score+=15
            if fact:score+=5
            ident=hashlib.sha256(f"{page['document_id']}:{page['page']}:{text}".encode()).hexdigest()[:24]
            candidates.append({'id':ident,**{k:v for k,v in page.items() if k!='text'},'text':text,
                               'match_score':round(score,3),'retrieval':'contextual local search'})
    candidates.sort(key=lambda c:(-c['match_score'],c['filename'],c['page']))
    # Include each policy separately instead of silently choosing one employer.
    grouped=defaultdict(list)
    for c in candidates:
        group=grouped[c['document_id']]
        if len(group)<2 and not any(compact(c['text']) in compact(x['text']) or compact(x['text']) in compact(c['text']) for x in group):group.append(c)
    selected=[]
    for position in range(2):
        for group in grouped.values():
            if position<len(group) and len(selected)<8:selected.append(group[position])
    return selected


def extract_answer(question,citations):
    wanted=topics_in(question)
    if not wanted and compact(question) in {'ลา','การลา','วันลา','ลาได้กี่วัน','leave','leavedays'}:
        return 'กรุณาระบุประเภทการลา เช่น ลาป่วย ลากิจ หรือลาพักร้อน เพื่อค้นสิทธิและเงื่อนไขให้ตรงคำถาม'
    if not citations:
        if wanted:
            return 'ยังไม่พบรายละเอียดที่ตอบคำถามได้ในข้อความที่อ่านจากเอกสาร จึงไม่สามารถระบุเวลา หรือจำนวนวันแทนเอกสารได้ กรุณาเลือกคู่มือที่ต้องการหรือตรวจว่าหน้ารายละเอียดอ่านเป็นข้อความได้'
        return 'ไม่พบรายละเอียดที่ตรงคำถามในเอกสาร กรุณาระบุหัวข้อให้ชัดเจน'
    heading='ข้อมูลตามเอกสาร (คัดข้อความพร้อมเงื่อนไข ไม่ใช่คำตอบที่สร้างโดย AI):'
    parts=[heading]
    for c in citations:
        parts.append(f"{c['filename']} · หน้า {c['page']} [{c['reference']}]\n{c['text']}")
    if len({c['document_id'] for c in citations})>1:
        parts.insert(1,'พบหลายคู่มือ ข้อมูลด้านล่างแยกตามไฟล์ โปรดเลือกคู่มือของหน่วยงานที่ต้องการ')
    return '\n\n'.join(parts)
