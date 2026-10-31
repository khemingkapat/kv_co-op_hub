# 🚀 คู่มือการใช้งาน KMUTT CPE Slurm GPU Cluster (Step-by-Step Guide)

> **วิชา/โครงงาน:** CPE401 Senior Project I & CPE393 GPU Programming  
> **ภาควิชาวิศวกรรมคอมพิวเตอร์ มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าธนบุรี (KMUTT)**  
> **Server Host:** `portal.slurm.cpe.kmutt.ac.th` | **Login Node:** `ctl01`  
> **GPU Hardware:** NVIDIA GeForce RTX 4090 (24 GB GDDR6X)

---

## 📌 สารบัญ (Table of Contents)
1. [เตรียมตัวก่อนเริ่ม (Prerequisites & VPN)](#1-เตรียมตัวก่อนเริ่ม-prerequisites--vpn)
2. [วิธีการเชื่อมต่อเข้า Server (2 รูปแบบ)](#2-วิธีการเชื่อมต่อเข้า-server-2-รูปแบบ)
3. [กฎเหล็กและโครงสร้างของ Slurm Cluster](#3-กฎเหล็กและโครงสร้างของ-slurm-cluster)
4. [คำสั่ง Slurm ที่ต้องใช้บ่อยที่สุด (Cheat Sheet)](#4-คำสั่ง-slurm-ที่ต้องใช้บ่อยที่สุด-cheat-sheet)
5. [โครงสร้างไฟล์ `.sbatch` และตัวอย่างสคริปต์รันจริง](#5-โครงสร้างไฟล์-sbatch-และตัวอย่างสคริปต์รันจริง)
6. [ปัญหาที่พบบ่อยและวิธีแก้ (Troubleshooting & FAQs)](#6-ปัญหาที่พบบ่อยและวิธีแก้-troubleshooting--faqs)

---

## 1. เตรียมตัวก่อนเริ่ม (Prerequisites & VPN)

### 🔑 1.1 เชื่อมต่อ VPN มหาวิทยาลัย (กรณีอยู่นอก มจธ.)
หากไม่ได้เชื่อมต่อ Wi-Fi มหาวิทยาลัย (`@KMUTT-Student` หรือสาย LAN ในแล็บ) **จะต้องเปิด VPN เสมอ** มิฉะนั้นจะเชื่อมต่อ SSH ไม่ติด (`Connection timed out`):
- เปิดโปรแกรม **KMUTT VPN (Cisco AnyConnect / FortiClient / KMUTT VPN)**
- เชื่อมต่อให้เรียบร้อยจนขึ้นสถานะ **Connected**

### ⚙️ 1.2 ตรวจสอบไฟล์ SSH Config ในเครื่องตัวเอง
ไฟล์ตั้งค่าจะอยู่ที่ `C:\Users\<Username>\.ssh\config` (บน Windows) ควรมีคอนฟิกดังนี้:

```ssh
Host slurm-portal
    HostName portal.slurm.cpe.kmutt.ac.th
    User 66070503456@cpe.kmutt.ac.th

Host slurm-direct
    HostName portal.slurm.cpe.kmutt.ac.th
    User 66070503456@cpe.kmutt.ac.th
```

---

## 2. วิธีการเชื่อมต่อเข้า Server (2 รูปแบบ)

### 🌟 วิธีที่ 1: ผ่าน VS Code Remote-SSH (แนะนำที่สุด - สะดวก มีหน้าต่างไฟล์)
1. เปิด **Visual Studio Code**
2. กดคีย์ลัด: `Ctrl + Shift + P` (หรือ `Cmd + Shift + P` บน Mac)
3. พิมพ์ค้นหา: **`Remote-SSH: Connect to Host...`** แล้วกด Enter
4. เลือก **`slurm-portal`**
5. หากระบบถาม Platform ให้เลือก **`Linux`**
6. กรอก **รหัสผ่าน (Password)** บัญชีนักศึกษา KMUTT
7. เมื่อมุมซ้ายล่างขึ้นสถานะสีน้ำเงิน/เขียว `SSH: slurm-portal` แสดงว่าเชื่อมต่อสำเร็จ!
8. กดที่เมนู **File → Open Folder** เพื่อเปิดโฟลเดอร์งานบน Server เช่น `/storage/slurm/home/66070503456@cpe.kmutt.ac.th/cpe401-baseline`

---

### 💻 วิธีที่ 2: ผ่าน PowerShell / Terminal โดยตรง
เปิด PowerShell ในเครื่องคอมพิวเตอร์ของคุณ แล้วรัน:

```powershell
ssh slurm-portal
# หรือรันแบบเต็ม:
ssh 66070503456@cpe.kmutt.ac.th@portal.slurm.cpe.kmutt.ac.th
```
ใส่ Password แล้วจะเข้าสู่หน้าต่าง Command Line ของ `ctl01`:
```text
66070503456@cpe.kmutt.ac.th@ctl01:~$ 
```

---

## 3. กฎเหล็กและโครงสร้างของ Slurm Cluster

ระบบ Cluster แบ่งออกเป็น 2 ส่วนหลัก:

```text
[เครื่องคอมพิวเตอร์ของคุณ]
         │ (SSH ผ่าน VPN)
         ▼
[Login Node: ctl01]  <--- คุณยืนอยู่ที่นี่! (ห้ามรันงานหนัก/โมเดลที่นี่)
         │
         │ (ยื่นใบคำสั่งงานด้วยคำสั่ง sbatch)
         ▼
[Slurm Workload Manager]  <--- ผู้จัดคิวและกระจายงาน
         │
         ▼
[Compute/GPU Node: เช่น gpu01]  <--- เครื่องจริงที่มีการ์ดจอ RTX 4090
         │ (รันงานเสร็จแล้ว)
         ▼
[บันทึกผลลงไฟล์: slurm-<JobID>.log]
```

> [!CAUTION]
> **กฎเหล็กสำคัญที่สุด (Golden Rule):**  
> เครื่อง `ctl01` ที่เราล็อกอินเข้าไปคือ **Login Node** ซึ่งเป็นเครื่องแชร์ทรัพยากรกลางของนักศึกษาทุกคน **ห้ามรันโปรแกรม AI, รันโมเดล หรือเทรนโค้ดบน `ctl01` โดยตรงเด็ดขาด!** ต้องเขียนสคริปต์แล้วส่งงานผ่านคำสั่ง **`sbatch`** เท่านั้น เพื่อให้ระบบไปดึงการ์ดจอจากเครื่อง Compute Node มาให้เรา

---

## 4. คำสั่ง Slurm ที่ต้องใช้บ่อยที่สุด (Cheat Sheet)

| คำสั่ง | หน้าที่การทำงาน | ตัวอย่างการใช้งาน |
| :--- | :--- | :--- |
| **`sbatch <script>.sbatch`** | ส่งงาน (Job) เข้าคิวในระบบ Slurm | `sbatch run_baseline_slurm.sbatch` |
| **`squeue -u $USER`** | ตรวจดูสถานะงานของตัวเองว่ารันอยู่หรือยัง | `squeue -u $USER` |
| **`tail -f slurm-<JobID>.log`** | ดูผลลัพธ์การทำงานแบบ **Real-time (สดๆ)** | `tail -f slurm-22091.log` *(กด Ctrl+C เพื่อออก)* |
| **`cat $(ls -t slurm-*.log \| head -1)`** | เปิดอ่าน Log ของ **งานรอบล่าสุด** (ไม่ต้องจำเลข Job ID) | `cat $(ls -t slurm-*.log \| head -1)` |
| **`cat slurm-<JobID>.err`** | เปิดดูข้อความ Error หรือ Warning ของงานนั้น | `cat slurm-22091.err` |
| **`scancel <JobID>`** | สั่งยกเลิก/หยุดงานที่กำลังรันทันที | `scancel 22091` |
| **`sinfo`** | ตรวจสอบสถานะของทุกโหนดและการ์ดจอในระบบ | `sinfo` |

### 💡 ทริก: อยากเปิด Terminal สดบนเครื่องที่มีการ์ดจอ (Interactive GPU Shell)
หากต้องการต่อเข้าไปลองรันคำสั่งสด ๆ บนเครื่องการ์ดจอ (ไม่ใช่ Login Node):
```bash
srun --partition=gpu4090 --gres=gpu:rtx4090:1 --cpus-per-task=4 --mem=16G --pty bash
```
เมื่อคำสั่งทำงาน สังเกตชื่อเครื่องจะเปลี่ยนจาก `ctl01` กลายเป็น `gpu01` ทันที สามารถรัน `nvidia-smi` และทดสอบโค้ดได้สด ๆ เมื่อเสร็จแล้วพิมพ์ `exit` เพื่อกลับมายัง Login Node

---

## 5. โครงสร้างไฟล์ `.sbatch` และตัวอย่างสคริปต์รันจริง

ในการส่งงานเข้า Slurm เราจะสร้างไฟล์ Shell Script ที่มีนามสกุล `.sbatch` ดังนี้:

### ตัวอย่างไฟล์: `run_baseline_slurm.sbatch`
```bash
#!/bin/bash
#SBATCH --job-name=kv_baseline_profile   # ตั้งชื่องาน
#SBATCH --output=slurm-%j.log            # ไฟล์ Output Log (%j จะแทนที่ด้วย Job ID)
#SBATCH --error=slurm-%j.err             # ไฟล์ Error Log
#SBATCH --partition=gpu4090              # ระบุ Partition การ์ดจอ (เช่น gpu4090)
#SBATCH --gres=gpu:rtx4090:1             # ขอการ์ดจอ RTX 4090 จำนวน 1 ใบ
#SBATCH --cpus-per-task=8                # ขอ CPU 8 คอร์
#SBATCH --mem=32G                        # ขอ RAM 32 GB
#SBATCH --time=02:00:00                  # เวลาทำงานสูงสุด (ชั่วโมง:นาที:วินาที)

# โหลด Environment มาตรฐาน
source /etc/profile
set -euo pipefail

# 1. โหลด CUDA Toolkit
module purge
module load cuda/12.2

# 2. เช็กสถานะการ์ดจอที่ได้รับจัดสรร
nvidia-smi

# 3. เปิดใช้งาน Python Virtual Environment
source "$HOME/.venv_kv_baseline/bin/activate"

# 4. คำสั่งรันโปรแกรม Python ของเรา
python profile_baseline.py
```

---

## 6. ปัญหาที่พบบ่อยและวิธีแก้ (Troubleshooting & FAQs)

### ❓ 1. เชื่อมต่อ SSH ไม่ติด (`Connection timed out`)
- **สาเหตุ:** ยังไม่ได้เชื่อมต่อ KMUTT VPN หรืออินเทอร์เน็ตหลุด
- **วิธีแก้:** เปิดโปรแกรม VPN ของมหาวิทยาลัย แล้วลองเชื่อมต่อใหม่อีกครั้ง

### ❓ 2. สั่ง `sbatch` แล้วขึ้น `sbatch: error: Unable to open file ...`
- **สาเหตุ:** ไฟล์ `.sbatch` ยังอยู่ในเครื่องคอมพิวเตอร์ของคุณ ยังไม่ได้ถูกอัปโหลดหรือสร้างขึ้นบน Server
- **วิธีแก้:** ตรวจสอบโฟลเดอร์ปัจจุบันด้วย `pwd` และ `ls` หรือสร้างไฟล์บน Server ผ่าน VS Code ก่อนสั่งรัน

### ❓ 3. สั่ง `squeue -u $USER` แล้วสถานะขึ้น `PD` (Pending) นานมาก
- **สาเหตุ:** คิวการ์ดจอเต็ม กำลังรอคิวให้ผู้ใช้คนอื่นรันเสร็จ
- **วิธีแก้:** ดูเหตุผลในช่อง `NODELIST(REASON)` ถ้าขึ้น `(Priority)` หรือ `(Resources)` ให้รอระบบจัดสรร หรือตรวจดูว่า partition ที่ขอถูกต้องหรือไม่

### ❓ 4. เปิดดู Log แล้วเจอข้อความเดิมของรอบก่อนหน้า
- **สาเหตุ:** เผลอไปเปิดไฟล์ log ของ Job ID เก่า (เช่น `cat slurm-22089.log`)
- **วิธีแก้:** เช็กเลข Job ID ใหม่เสมอ หรือใช้คำสั่งเปิด log ล่าสุด:
  ```bash
  cat $(ls -t slurm-*.log | head -1)
  ```

### ❓ 5. Warning ในไฟล์ `.err` ถือว่าเป็น Error หรือไม่?
- ข้อความจำพวก `FutureWarning`, `UserWarning: do_sample is set to False`, หรือหลอดดาวน์โหลด `Loading checkpoint shards: 100%` **ไม่ใช่ Error** ครับ เป็นเพียงคำเตือนทั่วไปของ Python และหลอด Progress bar ที่แสดงผลออกทางช่อง stderr ถือว่ารันสำเร็จสมบูรณ์ตามปกติ

---

*เอกสารทางการเพิ่มเติมของภาควิชา: [wiki.slurm.cpe.kmutt.ac.th](http://wiki.slurm.cpe.kmutt.ac.th)*
