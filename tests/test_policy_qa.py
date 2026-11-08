from policy_qa import retrieve, extract_answer, page_chunks, reconstruct_pages


def rows_from_pages(pages):
    rows={'ids':[],'documents':[],'metadatas':[]}
    for docid,name,page,text in pages:
        for i,line in enumerate(text.splitlines(),1):
            rows['ids'].append(f'{docid}-{i}')
            rows['documents'].append(line)
            rows['metadatas'].append({'document_id':docid,'filename':name,'page':page})
    return rows


def answer(question, rows, docid=None):
    citations=retrieve(question,rows,docid)
    for i,c in enumerate(citations,1): c['reference']=f'Ref {i}'
    return extract_answer(question,citations), citations


def test_hours_retrieves_details_instead_of_toc_or_heading():
    rows=rows_from_pages([
        ('a','Manual_A.pdf',2,'สารบัญ\nเวลาปฏิบัติงาน .... 5\nการลา .... 10'),
        ('a','Manual_A.pdf',5,'เวลาปฏิบัติงาน\nวันจันทร์ถึงวันศุกร์\nเวลา 08.30 - 17.30 น.\nพักกลางวัน 12.00 - 13.00 น.\nลาป่วย\nลาป่วยตามที่ป่วยจริง\nได้รับค่าจ้างไม่เกิน 30 วันทำงานต่อปี'),
    ])
    text,citations=answer('เวลาปฏิบัติงาน',rows)
    assert '08.30 - 17.30' in text
    assert '12.00 - 13.00' in text
    assert 'ลาป่วย' not in text
    assert all(c['page']==5 for c in citations)


def test_leave_preserves_paid_day_distinction_and_conditions():
    rows=rows_from_pages([('a','Manual_A.pdf',10,'ลาป่วย\nลาได้ตามที่ป่วยจริง\nได้รับค่าจ้างไม่เกิน 30 วันทำงานต่อปี\nลาตั้งแต่ 3 วันทำงานต้องแสดงใบรับรองแพทย์\nลากิจ\nลาได้ 6 วันต่อปี')])
    text,citations=answer('ลาป่วยได้กี่วัน',rows)
    assert 'ตามที่ป่วยจริง' in text and 'ค่าจ้างไม่เกิน 30' in text and '3 วัน' in text
    assert '6 วัน' not in text
    assert citations


def test_two_employers_are_separate_and_filter_works():
    rows=rows_from_pages([
        ('a','Manual_A.pdf',5,'เวลาปฏิบัติงาน\n08.00 - 17.00 น.'),
        ('b','Manual_B.pdf',6,'เวลาปฏิบัติงาน\n09.00 - 18.00 น.'),
    ])
    text,citations=answer('เวลาทำงาน',rows)
    assert 'Manual_A.pdf' in text and 'Manual_B.pdf' in text
    assert '08.00' in text and '09.00' in text
    only,_=answer('เวลาปฏิบัติงาน',rows,'b')
    assert '09.00' in only and '08.00' not in only


def test_heading_without_facts_never_invents_hours_or_days():
    rows=rows_from_pages([('a','Manual_A.pdf',2,'เวลาปฏิบัติงาน\nการลา\nลาป่วย')])
    text,citations=answer('เวลาปฏิบัติงาน',rows)
    assert not citations and 'ยังไม่พบรายละเอียด' in text
    assert '08.00' not in text
    text,citations=answer('ลาป่วย',rows)
    assert not citations and '30' not in text


def test_reconstruct_new_overlapping_chunks_exactly():
    text=('รายละเอียดนโยบายพนักงาน\n'*100)+'เวลาปฏิบัติงาน\n08.30 - 17.30 น.'
    rows={'ids':[],'documents':[],'metadatas':[]}
    for i,(piece,offset) in enumerate(page_chunks(text),1):
        rows['ids'].append(f'a-{i}');rows['documents'].append(piece)
        rows['metadatas'].append({'document_id':'a','filename':'Manual_A.pdf','page':1,'index_version':2,'offset':offset})
    assert reconstruct_pages(rows)[0]['text']==text
    response,citations=answer('เวลาปฏิบัติงาน',rows)
    assert '08.30 - 17.30' in response


def test_english_synonym_searches_thai_policy_and_generic_leave_clarifies():
    rows=rows_from_pages([('a','Manual_A.pdf',4,'เวลาปฏิบัติงาน\nเวลา 08.30 - 17.30 น.')])
    text,citations=answer('What are the working hours?',rows)
    assert '08.30' in text
    text,_=answer('ลาได้กี่วัน',rows)
    assert 'ระบุประเภทการลา' in text
