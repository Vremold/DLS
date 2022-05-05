import os
import re
import time
import json
import shutil

import markdown
from bs4 import BeautifulSoup

from config import MIN_STARS
from pathutil import RAW_GH_REPOSITORIES_FILE, RAW_GH_REPOSITORY_READEME_DIR, CLEANED_GH_REPOSITORY_FILE, CLEANED_GH_REPOSITORY_README_DIR

class RepoFilter(object):
    def __init__(self):
        pass

    def generate_readme_identifier_from_gh_url(self, url):
        idx = url.rfind("/")
        repo_name = url[idx+1:]
        idx2 = url[:idx].rfind("/")
        repo_owner = url[idx2 + 1: idx]
        return "{}#{}.md".format(repo_owner, repo_name)
    
    def filter_repo(self, src_repositories_file, dst_repositories_file):
        selected_readme_identifiers = set()
        with open(src_repositories_file, "r", encoding="utf-8") as inf, \
            open(dst_repositories_file, "w", encoding="utf-8") as outf:
            for line in inf:
                obj = json.loads(line)
                if obj["git_stars"] >= MIN_STARS:
                    outf.write(line)
                    selected_readme_identifiers.add(self.generate_readme_identifier_from_gh_url(obj["git_url"]))
        return selected_readme_identifiers

class ReadmeCleaner(object):
    def __init__(self):
        pass

    def clean_raw_readme_txt(self, text):
        # remove blank lines
        text = re.sub(r"^\s*$\n", "", text)
        # remove code_blocks
        text = re.sub(r"```[\s\S]*?```", "", text)
        # remove badges
        pattern = r"\[!\[.*?\]\(.*?\)\]\(.*?\)"
        text = re.sub(pattern, "", text)
        pattern = r"!\[.*?\]\(.*?\)"
        text = re.sub(pattern, "", text)
        # remove hyperlinks
        text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
        return text
    
    def convert_md_to_html(self, readme_text):
        return markdown.markdown(readme_text)

    def get_html_element_content(self, name, element):
        if name in ["h1", "h2", "h3", "h4", "h5"]:
            return element.get_text()
        elif name == "a" or name == "p":
            return element.get_text()
        elif name == "ul":
            texts = []
            for li in element.find_all("li"):
                texts.append(li.get_text())
            return texts
        else:
            return ""
    
    def get_html_content_before_head(self, soup):
        p = soup.find("p")
        if not p:
            return []
        sib = p
        contents = [self.get_html_element_content("p", sib)]
        while True:
            sib = sib.find_next_sibling()
            if not sib:
                return contents
            if sib.name in ["h1", "h2", "h3", "h4", "h5"]:
                return contents
            contents.append(self.get_html_element_content(sib.name, sib))
    
    def get_html_head_content(self, head):
        contents = []
        sib = head
        if not sib:
            return None, "", "", []
        while True:
            sib = sib.find_next_sibling()
            if not sib:
                return None, head.name, head.get_text(), contents
            if sib.name in ["h1", "h2", "h3", "h4", "h5"]:
                return sib, head.name, head.get_text(), contents
            else:
                contents.append(self.get_html_element_content(sib.name, sib))

    def parse_html_and_save(self, html_text, dst_readme_file):
        # print(html_text)
        outf = open(dst_readme_file, "w", encoding="utf-8")
        soup = BeautifulSoup(html_text, "html.parser")

        # content before the first head title
        next_head = soup.find(["h1", "h2", "h3", "h4", "h5"])
        if next_head and next_head.find_previous_sibling():
            contents = self.get_html_content_before_head(soup)
            outf.write("{}\t{}\t{}\n".format("h", "", json.dumps(contents)))
        
        # 按照标题将剩下的文档分割
        next_head, name, head_text, contents = self.get_html_head_content(next_head)
        while next_head is not None:
            outf.write("{}\t{}\t{}\n".format(name, head_text, json.dumps(contents)))
            next_head, name, head_text, contents = self.get_html_head_content(next_head)
        outf.write("{}\t{}\t{}\n".format(name, head_text, json.dumps(contents)))

    def clean(self, src_readme_file, dst_readme_file):
        with open(src_readme_file, "r", encoding="utf-8") as inf:
            readme_text = inf.read()

        readme_text = self.clean_raw_readme_txt(readme_text)
        html_text = self.convert_md_to_html(readme_text)
        self.parse_html_and_save(html_text, dst_readme_file)

if __name__ == "__main__":
    repo_filter = RepoFilter()
    selected_readme_identifiers = repo_filter.filter_repo(RAW_GH_REPOSITORIES_FILE, CLEANED_GH_REPOSITORY_FILE)

    readme_cleaner = ReadmeCleaner()
    for filename in os.listdir(RAW_GH_REPOSITORY_READEME_DIR):
        if filename in selected_readme_identifiers:
            readme_cleaner.clean(
                os.path.join(RAW_GH_REPOSITORY_READEME_DIR, filename), 
                os.path.join(CLEANED_GH_REPOSITORY_README_DIR, filename[:-3]+".txt"))