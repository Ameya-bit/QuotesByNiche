import requests, time, re, pathlib

urls = {
    "twilight_of_the_idols": "https://www.gutenberg.org/cache/epub/52263/pg52263.txt",
    "genealogy_of_morals":   "https://www.gutenberg.org/cache/epub/52319/pg52319.txt",
    "zarathustra":           "https://www.gutenberg.org/cache/epub/1998/pg1998.txt",
    "beyond_good_and_evil":  "https://www.gutenberg.org/cache/epub/4363/pg4363.txt",
}

def strip_pg_boilerplate(text):
    start = re.search(r"\*\*\* START OF TH[EI]S? PROJECT GUTENBERG.*?\*\*\*", text)
    end   = re.search(r"\*\*\* END OF TH[EI]S? PROJECT GUTENBERG.*?\*\*\*", text)
    s = start.end() if start else 0
    e = end.start() if end else len(text)
    return text[s:e].strip()

cache = pathlib.Path("data"); cache.mkdir(exist_ok=True)
headers = {"User-Agent": "personal-ml-learning-project"}

with open(cache / "grabbed_nietzsche.txt", "w", encoding="utf-8") as out:
    for name, url in urls.items():
        f = cache / f"{name}.txt"
        if f.exists():
            body = f.read_text(encoding="utf-8")
        else:
            resp = requests.get(url, headers=headers)
            resp.raise_for_status()
            resp.encoding = "utf-8"
            body = strip_pg_boilerplate(resp.text)
            f.write_text(body, encoding="utf-8")
            time.sleep(2)  # be polite
        out.write(body + "\n\n")