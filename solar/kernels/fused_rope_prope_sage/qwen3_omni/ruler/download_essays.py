"""Same logic as RULER's scripts/data/synthetic/json/download_paulgraham_essay.py,
but with per-URL retries (the proxy occasionally resets connections, and the
upstream script silently drops failed essays). Writes PaulGrahamEssays.json into
the RULER clone. Usage: python -I download_essays.py <RULER>/scripts/data/synthetic/json
"""
import glob, json, os, shutil, sys, time, urllib.request
import html2text
from bs4 import BeautifulSoup

json_dir = sys.argv[1]
repo_dir = os.path.join(json_dir, 'essay_repo'); html_dir = os.path.join(json_dir, 'essay_html')
os.makedirs(repo_dir, exist_ok=True); os.makedirs(html_dir, exist_ok=True)
h = html2text.HTML2Text()
h.ignore_images = True; h.ignore_tables = True; h.escape_all = True
h.reference_links = False; h.mark_code = False

urls = [l.strip() for l in open(os.path.join(json_dir, 'PaulGrahamEssays_URLs.txt')) if l.strip()]
failed = []
for url in urls:
    is_html = '.html' in url
    filename = url.split('/')[-1].replace('.html', '.txt') if is_html else url.split('/')[-1]
    out = os.path.join(html_dir if is_html else repo_dir, filename)
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=60) as w:
                raw = w.read()
            if is_html:
                content = raw.decode("unicode_escape", "utf-8")
                parsed = h.handle(str(BeautifulSoup(content, 'html.parser').find('font')))
            else:
                parsed = raw.decode('utf-8')
            with open(out, 'w') as f:
                f.write(parsed)
            break
        except Exception as e:
            print(f"retry {attempt} {filename}: {e}", flush=True); time.sleep(2 + 2 * attempt)
    else:
        failed.append(filename)

files_repo = sorted(glob.glob(os.path.join(repo_dir, '*.txt')))
files_html = sorted(glob.glob(os.path.join(html_dir, '*.txt')))
print(f'Downloaded {len(files_repo)} repo essays + {len(files_html)} html essays; failed: {failed}')
text = "".join(open(f).read() for f in files_repo + files_html)
with open(os.path.join(json_dir, 'PaulGrahamEssays.json'), 'w') as f:
    json.dump({"text": text}, f)
shutil.rmtree(repo_dir); shutil.rmtree(html_dir)
if failed:
    sys.exit(1)
