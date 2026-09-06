import os
import re
import shutil
import zipfile
import asyncio
import tempfile
import logging
from pathlib import Path

import discord
from discord.ext import commands
import google.generativeai as genai
from PIL import Image
import gdown

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("manhwa_ocr_bot")

DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not DISCORD_TOKEN or not GEMINI_API_KEY:
    raise RuntimeError("المتغيرات البيئية غير مكتملة.")

genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL_NAME = "gemini-2.5-flash"
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

OCR_PROMPT = """
أنت أداة استخراج نصوص (OCR) متخصصة في صور المانهوا الكورية (Manhwa/Webtoon).

مهمتك:
استخرج فقط النصوص الكورية (Hangul) الأصلية الموجودة داخل هذه الصورة، بدون أي ترجمة إلى الإنجليزية أو العربية أو أي لغة أخرى. النص يجب أن يبقى بالكورية الأصلية كما هو مكتوب في الصورة تماماً.

رتب النصوص بالترتيب من الأعلى إلى الأسفل حسب ظهورها في الصورة (ثم من اليمين لليسار إن وجد نصوص بنفس المستوى الأفقي، حسب تدفق القراءة الطبيعي للمانهوا).

يجب عليك الالتزام الإجباري بالرموز التالية حسب نوع كل فقاعة أو سياق النص:

1. فقاعة كلام عادية (Speech Bubble): ضع النص بين علامتي تنصيص
   مثال: "안녕하세요"

2. فقاعة تفكير (Thought Bubble): ضع النص بين قوسين
   مثال: (이게 뭐지?)

3. فقاعة صراخ / صوت مرتفع (Shout Bubble): ضع نقطتين رأسيتين قبل النص مباشرة (بدون علامات تنصيص)
   مثال: :아아아악!

4. فقاعة نظام / نافذة نظام (System Window - كما في قصص الرجوع/الشحن): ضع النص بين علامتي أكبر من وأصغر من
   مثال: <레벨이 상승했습니다>

5. كلام الراوي (Narration / Text Box خارج الفقاعات): ضع الاختصار OT: قبل النص
   مثال: OT: 그날 이후로 모든 것이 변했다

6. كلام جانبي صغير تقوله الشخصية خارج الفقاعة الرسمية (SFX كلامي صغير، تعليق جانبي، همس مكتوب بخط صغير): ضع الاختصار ST: قبل النص
   مثال: ST: 헐...

قواعد صارمة يجب اتباعها:
- لا تترجم أي نص إطلاقاً. النص المطلوب هو الكوري الأصلي فقط.
- لا تضف أي شرح أو تعليق أو وصف للصورة.
- لا تكتب أي شيء عن الفقاعات نفسها (مثل "الفقاعة الأولى تحتوي على...")، فقط النص مع رمزه مباشرة.
- كل نص في سطر منفصل.
- إذا لم تجد أي نص كوري في الصورة، اكتب فقط: [لا يوجد نص]
- لا تخترع نصاً غير موجود في الصورة.
- التزم حرفياً بالرموز الستة أعلاه دون استثناء لكل سطر تستخرجه.

ابدأ الاستخراج الآن.
"""

def natural_sort_key(path: Path):
    name = path.stem
    return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in re.split(r"(\d+)", name)]

def download_zip_file_gdown(url: str, destination_path: Path) -> None:
    output = gdown.download(url=url, output=str(destination_path), quiet=True, fuzzy=True)
    if not output or not destination_path.exists() or destination_path.stat().st_size == 0:
        raise ValueError("فشل تحميل الملف. تأكد من أن الرابط صالح وأن الملف عام (Public) في Google Drive.")

def extract_zip_and_get_sorted_images(zip_path: Path, extract_dir: Path) -> list[Path]:
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    image_paths = [
        p for p in extract_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        and not p.name.startswith("__MACOSX")
        and "__MACOSX" not in str(p)
    ]
    image_paths.sort(key=natural_sort_key)
    return image_paths

def run_gemini_ocr_on_image(image_path: Path) -> str:
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    try:
        image = Image.open(image_path)
        image.load()
    except Exception as e:
        return "[تعذّر فتح الصورة]"

    try:
        response = model.generate_content([OCR_PROMPT, image], generation_config=genai.types.GenerationConfig(temperature=0.1))
        text = (response.text or "").strip()
        return text if text else "[لا يوجد نص]"
    except Exception as e:
        return f"[خطأ أثناء استخراج هذه الصفحة: {e}]"

async def process_all_pages(image_paths: list[Path], status_msg: discord.Message) -> str:
    loop = asyncio.get_event_loop()
    all_pages_text = []
    total = len(image_paths)

    for index, image_path in enumerate(image_paths, start=1):
        if index == 1 or index % 3 == 0 or index == total:
            try:
                await status_msg.edit(
                    content=f"🔎 جاري استخراج النصوص من **{total}** صفحة عبر Gemini...\n"
                            f"⏳ التقدم الحالي: الصفحة **{index}/{total}** (`{image_path.name}`)"
                )
            except Exception:
                pass

        page_text = await loop.run_in_executor(None, run_gemini_ocr_on_image, image_path)
        section = f"===== الصفحة {index:03d} ({image_path.name}) =====\n{page_text}\n"
        all_pages_text.append(section)
        await asyncio.sleep(0.5)

    return "\n".join(all_pages_text)

def cleanup_paths(*paths: Path) -> None:
    for p in paths:
        try:
            if p is None: continue
            if p.is_dir(): shutil.rmtree(p, ignore_errors=True)
            elif p.is_file(): p.unlink(missing_ok=True)
        except Exception: pass

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)

@bot.event
async def on_ready():
    logger.info(f"تم تسجيل الدخول بنجاح باسم: {bot.user}")

@bot.command(name="استخراج")
async def extract_command(ctx: commands.Context, url: str = None):
    if not url:
        await ctx.reply("⚠️ الرجاء إرفاق رابط Google Drive أو رابط مباشر لملف zip.")
        return

    status_message = await ctx.reply("⏳ جاري تنزيل الملف المضغوط...")
    work_dir = Path(tempfile.mkdtemp(prefix="manhwa_ocr_"))
    zip_path = work_dir / "chapter.zip"
    extract_dir = work_dir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    output_txt_path = work_dir / "extracted_korean_text.txt"

    try:
        await asyncio.get_event_loop().run_in_executor(None, download_zip_file_gdown, url, zip_path)
        await status_message.edit(content="📂 تم التنزيل! جاري فك الضغط وترتيب الصفحات...")
        image_paths = await asyncio.get_event_loop().run_in_executor(None, extract_zip_and_get_sorted_images, zip_path, extract_dir)

        if not image_paths:
            await status_message.edit(content="❌ لم يتم العثور على أي صور مدعومة داخل الملف.")
            cleanup_paths(work_dir)
            return

        final_text = await process_all_pages(image_paths, status_message)
        header = f"استخراج نصوص كورية - عدد الصفحات: {len(image_paths)}\n{'=' * 50}\n\n"
        output_txt_path.write_text(header + final_text, encoding="utf-8")

        await status_message.edit(content="📤 اكتمل الاستخراج! جاري رفع الملف النهائي...")
        discord_file = discord.File(output_txt_path, filename="extracted_korean_text.txt")
        await ctx.reply(content=f"✅ **تم الانتهاء بنجاح!** تم استخراج النصوص من {len(image_paths)} صفحة.", file=discord_file)
        await status_message.delete()

    except Exception as e:
        await status_message.edit(content=f"❌ حدث خطأ: {e}")
    finally:
        cleanup_paths(work_dir)

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
