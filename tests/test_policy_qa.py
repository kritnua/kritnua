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


def test_pdf_style_annual_leave_alias_and_incomplete_badges():
    rows=rows_from_pages([('a','Policy_A.pdf',3,'พักผ่อน 5 / 7 / 9 วัน\nลาป่วยรับค่าจ้าง 25 วัน\nลากิจจาเป็น 4 วัน\nหยุดประจาสัปดาห์อย่างน้อย 1 วัน\nพักผ่อนประจาปี (วันทางานต่อปี)\nอายุงาน 1–ไม่ครบ 3 ปี: 5 วัน\n3–ไม่ครบ 5 ปี: 7 วัน\n5 ปีขึ้นไป: 9 วัน\nขออนุมัติล่วงหน้าอย่างน้อย 2 วัน\nลาป่วยได้ตามที่ป่วยจริง ได้รับค่าจ้าง\nไม่เกิน 25 วันทางานต่อปี\nลาตั้งแต่ 3 วันอาจต้องแสดงใบรับรองแพทย์\nลากิจจาเป็น 4 วันทางานต่อปี\nรายเดือนได้รับค่าจ้าง 4 วัน\nรายวันได้รับ 3 วัน ตามเงื่อนไข')])
    annual,citations=answer('ลาพักร้อนกี่วัน',rows)
    assert '5 วัน' in annual and '7 วัน' in annual and '9 วัน' in annual
    assert 'ลาป่วย' not in annual
    sick,citations=answer('ลาป่วย',rows)
    assert len(citations)==1
    assert 'ตามที่ป่วยจริง' in sick and '25 วัน' in sick and 'ใบรับรองแพทย์' in sick
    personal,citations=answer('ลากิจ',rows)
    assert 'รายเดือน' in personal and 'รายวัน' in personal
    assert 'หยุดประจาสัปดาห์' not in personal


def test_spouse_childbirth_does_not_include_maternal_or_military_entitlements():
    rows=rows_from_pages([('a','Policy_A.pdf',4,'รับราชการทหาร\nได้รับค่าจ้างไม่เกิน 50 วัน\nลาคลอดบุตร\nลาได้ไม่เกิน 110 วัน\nบริษัทจ่ายค่าจ้าง 55 วัน\nการดูแลบุตรและคู่สมรส\nบุตรป่วยหรือมีภาวะตามเกณฑ์\nลาต่อจากลาคลอดได้ 10 วัน\nได้รับค่าจ้าง 50%\nคู่สมรสคลอดบุตร\nลาได้ 12 วัน พร้อมค่าจ้าง\nใช้ภายใน 80 วันนับแต่คลอด\nแยกจากสิทธิลากิจ\nการทางานระหว่างตั้งครรภ์\nแจ้ง HR เพื่อจัดงาน')])
    text,citations=answer('ภรรยาคลอดลาได้กี่วัน',rows)
    assert '12 วัน' in text and '80 วัน' in text and 'แยกจากสิทธิลากิจ' in text
    assert '110 วัน' not in text and '50 วัน' not in text
    maternal,_=answer('ลาคลอดบุตร',rows)
    assert '110 วัน' in maternal and '55 วัน' in maternal
    assert 'ลาได้ 12 วัน' not in maternal and 'ลาต่อจากลาคลอด' not in maternal


def test_ai_evidence_includes_small_manual_without_keyword_match():
    from policy_qa import ai_evidence
    rows=rows_from_pages([('a','Policy_A.pdf',4,'คู่สมรสคลอดบุตร\nลาได้ 12 วัน พร้อมค่าจ้าง'),('b','Policy_B.pdf',2,'เวลาปฏิบัติงาน\n09.00–18.00 น.')])
    citations,complete=ai_evidence('อยากอยู่ช่วยแฟนหลังมีน้อง',rows,'a')
    assert complete and len(citations)==1
    assert '12 วัน' in citations[0]['text']
    assert citations[0]['document_id']=='a'
