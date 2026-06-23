import requests, time, re, pathlib

urls = {
    "twilight_of_the_idols": "https://www.gutenberg.org/cache/epub/52263/pg52263.txt",
    "genealogy_of_morals":   "https://www.gutenberg.org/cache/epub/52319/pg52319.txt",
    "zarathustra":           "https://www.gutenberg.org/cache/epub/1998/pg1998.txt",
    "beyond_good_and_evil":  "https://www.gutenberg.org/cache/epub/4363/pg4363.txt",
    "joyful_wisdom": "https://www.gutenberg.org/cache/epub/52881/pg52881.txt",
    "human_all_too_human_1": "https://www.gutenberg.org/cache/epub/51935/pg51935.txt",
    "human_all_too_human_2": "https://www.gutenberg.org/cache/epub/37841/pg37841.txt",
    "dawn_of_day":        "https://www.gutenberg.org/cache/epub/39955/pg39955.txt",  # Daybreak
    "ecce_homo":          "https://www.gutenberg.org/cache/epub/52190/pg52190.txt",
    "birth_of_tragedy":   "https://www.gutenberg.org/cache/epub/51356/pg51356.txt",
    "will_to_power_1_2":  "https://www.gutenberg.org/cache/epub/52914/pg52914.txt",
    "will_to_power_3_4":  "https://www.gutenberg.org/cache/epub/52915/pg52915.txt",
    "case_of_wagner":     "https://www.gutenberg.org/cache/epub/25012/pg25012.txt",
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