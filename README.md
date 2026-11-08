# HR Policy Intelligence

สร้างใหม่จากภาพ Architecture ที่ผู้ใช้ให้ ไม่ใช่ซอร์สโค้ดต้นฉบับ

## สิ่งที่มี
- UI ตามภาพ: Sidebar สีน้ำเงินเข้ม, Architecture pipeline, Overview, Assistant, Knowledge Base, Evaluation
- FastAPI + Vanilla JS + ChromaDB
- อัปโหลด PDF/TXT ไม่เกิน 10 MB, แบ่งข้อความพร้อม filename/page/chunk ID
- Offline retrieval ด้วย deterministic feature hashing; แสดงข้อความที่ค้นพบโดยไม่อ้างว่าเป็นคำตอบจาก AI
- Gemini generation + embeddings เมื่อใส่ API key ใน server environment
- แสดงหลักฐานและตรวจว่า citation IDs อยู่ในผลค้นหา (ไม่รับรองความถูกต้องของข้ออ้าง)
- ชุดทดสอบ retrieval ตัวอย่าง 7 กรณี

## รันในเครื่อง
ต้องใช้ Python 3.11+

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```
เปิด http://localhost:8000/#architecture
เลือก Overview → Load sample policies เพื่อเริ่มทดลอง

## Deploy บน Render
1. แตก ZIP แล้วอัปโหลดไฟล์ทั้งหมดไปยัง GitHub repository ใหม่ โดยให้ app.py, requirements.txt และ render.yaml อยู่ระดับบนสุด
2. ลงชื่อเข้าใช้ https://dashboard.render.com/
3. เลือก New → Blueprint → เชื่อม GitHub repository นี้ → ตรวจสอบชื่อ service แล้ว deploy
4. ช่อง GEMINI_API_KEY ให้ใส่ key เฉพาะใน Render Environment หากต้องการ AI จริง ถ้าใช้ offline ให้เว้นว่างหรือลบตัวแปรนี้
5. เปิด URL ที่ Render ให้ เช่น https://hr-policy-intelligence-xxxx.onrender.com/#architecture

หรือเลือก New → Web Service → Python 3 แล้วตั้ง:
- Build: `pip install -r requirements.txt`
- Start: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- Health check: `/api/health`
- Python: 3.11.11

ชื่อ subdomain ขึ้นกับความพร้อมของ Render ไม่สามารถรับประกัน ai.onrender.com หรือใช้ชื่อเว็บเดิมได้

## ข้อมูลถาวร
ไฟล์ render.yaml ใช้ free plan และ DATA_DIR=/tmp/hr-policy-data: ข้อมูลอาจหายเมื่อ deploy/restart และบริการอาจพักเมื่อไม่มีคนใช้
สำหรับข้อมูลถาวร เปลี่ยนไปใช้ plan ที่รองรับ persistent disk เพิ่ม disk ที่ mount path `/var/data` แล้วตั้ง DATA_DIR=/var/data/hr-policy-data
ใช้ service instance เดียวสำหรับ local ChromaDB; การขยายหลาย instance ต้องใช้ shared vector service

## ตั้งค่า Gemini
- GEMINI_API_KEY: secret เก็บบน server เท่านั้น
- GEMINI_MODEL: gemini-2.5-flash (เปลี่ยนตามรุ่นที่บัญชีใช้งานได้)
- EMBEDDING_MODEL: gemini-embedding-001
เมื่อเปลี่ยน embedding provider/model จะใช้ collection ใหม่ ต้องอัปโหลดเอกสารใหม่
สถานะ "configured" หมายถึงมี key ไม่ได้แปลว่าทดสอบการเชื่อมต่อ Gemini สำเร็จแล้ว

## ขอบเขต
เป็น demo สำหรับเอกสารสังเคราะห์ ยังไม่มี login, การแบ่งสิทธิ์ผู้ใช้, rate limiting, OCR, policy versioning, Streamlit หรือผู้ให้บริการ OpenAI-compatible
ห้ามอัปโหลดข้อมูลพนักงานจริงใน deployment สาธารณะนี้ ก่อนใช้งานจริงควรเพิ่ม authentication และสิทธิ์เข้าถึงเอกสาร
โครงสร้างหน้า Architecture อ้างอิงภาพที่ให้ ส่วนหน้าอื่นสร้างใหม่ตามเมนู ไม่ได้ตรวจเทียบกับเว็บต้นฉบับ
ค่า similarity cutoff 0.15 เป็นค่าตัวอย่าง ต้อง calibrate ด้วยชุดทดสอบจริง
การตรวจ evaluation เป็นเพียงค้นคำในหลักฐาน ไม่ใช่การประเมินความถูกต้องของคำตอบ

## ตรวจสอบ
```bash
pytest -q
```
