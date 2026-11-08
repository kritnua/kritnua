"""Version-bound, manually specified checks against the two supplied manuals."""
import hashlib
from collections import defaultdict
from policy_qa import compact,reconstruct_pages

PROFILES={
    'manual_dugong_v1':{'filename':'Manual_Dugong.pdf','fingerprint':'4e79b26cd446cd251c28903f0889acfe78f3f98f597d4b3b81da97e880d19d85'},
    'manual_thaptaewan_v1':{'filename':'Manual_Thaptaewan.pdf','fingerprint':'597cd9ffea1a488697192800ea8c586b9a550c1b42a301772c056a43d34a3bd3'},
}
# Expected clauses are reviewed source facts, never inferred from current answers.
CASES=[
    {'id':'office_hours','question':'เวลาปฏิบัติงานสำนักงานและเวลาพัก','page':2,'clauses':[
        ('Working days','สำนักงาน: จันทร์–เสาร์'),('Office hours','08.00–17.00 น.'),('Lunch break','พัก 12.00–13.00 น.') ]},
    {'id':'service_hours','question':'เวลาปฏิบัติงานของงานบริการ','page':2,'clauses':[
        ('Shift schedule','งานบริการ: ปฏิบัติงานตามตารางกะ'),('Work days','ปกติ 6 วันต่อสัปดาห์'),('Hours per day','วันละไม่เกิน 8 ชั่วโมง'),('Hours per week','รวมไม่เกิน 48 ชั่วโมงต่อสัปดาห์')]},
    {'id':'sick_leave','question':'ลาป่วยได้กี่วันและมีเงื่อนไขอย่างไร','page':3,'clauses':[
        ('Actual illness entitlement','ลาป่วยได้ตามที่ป่วยจริง'),('Paid-day limit','ได้รับค่าจ้างไม่เกิน 30 วันทำงานต่อปี'),('Medical certificate condition','ลาตั้งแต่ 3 วันทำงาน อาจต้องแสดงใบรับรองแพทย์')]},
    {'id':'personal_leave','question':'ลากิจได้กี่วัน รายเดือนและรายวันได้รับค่าจ้างต่างกันอย่างไร','page':3,'clauses':[
        ('Leave entitlement','ลากิจจำเป็น 6 วันทำงานต่อปี'),('Monthly paid days','รายเดือนได้รับค่าจ้าง 6 วัน'),('Daily paid days','รายวันได้รับอย่างน้อย 3 วัน'),('Agreement for days 4–6','วันที่ 4–6 ตามข้อตกลง')]},
    {'id':'annual_leave','question':'ลาพักร้อนตามอายุงานและต้องแจ้งล่วงหน้ากี่วัน','page':3,'clauses':[
        ('Tenure 1–under 3 years','อายุงาน 1–ไม่ครบ 3 ปี: 6 วัน'),('Tenure 3–under 5 years','3–ไม่ครบ 5 ปี: 8 วัน'),('Tenure 5+ years','5 ปีขึ้นไป: 10 วัน'),('Advance notice','ขออนุมัติล่วงหน้าอย่างน้อย 3 วัน')]},
    {'id':'maternity_leave','question':'ลาคลอดบุตรได้กี่วันและได้รับค่าจ้างกี่วัน','page':4,'clauses':[
        ('Maternity entitlement','ลาได้ไม่เกิน 120 วันต่อครรภ์'),('Included days','รวมตรวจครรภ์และวันหยุดระหว่างลา'),('Paid-day limit','บริษัทจ่ายค่าจ้างไม่เกิน 60 วัน')]},
    {'id':'spouse_leave','question':'คู่สมรสคลอดบุตรลาได้กี่วัน ใช้เมื่อไร','page':4,'clauses':[
        ('Paid leave','ลาได้ไม่เกิน 15 วัน พร้อมค่าจ้าง'),('Use window','ใช้ภายใน 90 วันนับแต่คลอด'),('Separate entitlement','แยกจากสิทธิลากิจ')]},
    {'id':'childcare_leave','question':'บุตรป่วย ลาต่อจากลาคลอดได้กี่วันและรับค่าจ้างเท่าไร','page':4,'clauses':[
        ('Eligibility','บุตรป่วยหรือมีภาวะตามเกณฑ์'),('Additional days','ลาต่อจากลาคลอดได้ไม่เกิน 15 วัน'),('Pay and certificate','ได้รับค่าจ้าง 50% พร้อมใบรับรองแพทย์')]},
]

def normalized(text):
    return compact(text).replace('–','-').replace('—','-').replace('−','-')

def fingerprint(pages):
    value='\n'.join(f"{p['page']}:{compact(p['text'])}" for p in sorted(pages,key=lambda p:p['page']))
    return hashlib.sha256(value.encode()).hexdigest()

def assess(case,response,document):
    citations=response.get('citations',[])
    answer=normalized(response.get('answer',''))
    correct=[c for c in citations if c.get('document_id')==document['id'] and c.get('filename')==document['name'] and c.get('page')==case['page']]
    source=normalized('\n'.join(c['text'] for c in correct))
    checks=[{'label':label,'passed':normalized(expected) in answer and normalized(expected) in source,'expected':expected} for label,expected in case['clauses']]
    checks+=[{'label':'Correct document and filename','passed':bool(citations) and all(c.get('document_id')==document['id'] and c.get('filename')==document['name'] for c in citations)},
             {'label':f"Source page {case['page']}",'passed':bool(correct) and all(c.get('page')==case['page'] for c in citations)},
             {'label':'References attached to answer','passed':bool(correct) and all(f"[{c.get('reference')}]" in response.get('answer','') for c in correct)}]
    return {'id':case['id'],'question':case['question'],'filename':document['name'],'document_id':document['id'],
            'status':'PASS' if all(c['passed'] for c in checks) else 'FAIL','expected_page':case['page'],
            'checks':checks,'answer':response.get('answer',''),'citations':citations}

async def evaluate(rows,answer_fn,document_id=None):
    groups=defaultdict(list)
    for page in reconstruct_pages(rows):groups[page['document_id']].append(page)
    results=[];documents=[]
    for docid,pages in groups.items():
        if document_id and docid!=document_id:continue
        name=pages[0]['filename'];doc={'id':docid,'name':name}
        profile=next(((key,p) for key,p in PROFILES.items() if p['filename']==name),None)
        if profile is None:
            documents.append({**doc,'status':'NOT_CONFIGURED','reason':'No reviewed test baseline for this document. This is not a failed answer test.'})
            continue
        key,p=profile
        if fingerprint(pages)!=p['fingerprint']:
            documents.append({**doc,'status':'VERSION_MISMATCH','reason':'The document content or page layout differs from the reviewed manual. Review expected answers before testing.'})
            continue
        documents.append({**doc,'status':'TESTED','profile':key})
        for case in CASES:
            response=await answer_fn(case['question'],docid)
            results.append(assess(case,response,doc))
    passed=sum(r['status']=='PASS' for r in results)
    return {'results':results,'documents':documents,'summary':{'passed':passed,'failed':len(results)-passed,'total':len(results),'unverified_documents':sum(d['status']!='TESTED' for d in documents)},
            'scope':'local_policy_answers','note':'Checks local answers, policy conditions, document identity, page and citation linkage against reviewed PDF versions. Does not call AI or measure AI translation, legal validity, or general answer accuracy.'}
