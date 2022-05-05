"""
建图：
    Node:
        User:
        LibName:
        NE0-NE5:
        NPMDomain:
        GitHubTopic:
        NPMKeyWord:
    Edges:
        BelongToUser:
        HasDependence:
        HasDevDependence:
        RelateToNE:
        # HasDomain
        HasTopic
        HasKeyword
        SameToNE
"""
import os
import json
import sys

import networkx as nx 
from fuzzywuzzy import fuzz

from ..pathutil import CLEANED_GH_REPOSITORY_FILE

class NEAbbr(object):
    def __init__(self):
        abbrs = {
            "RAML": "RESTful API Modeling Language",
            "YAML": "Yet Another Markup Language",
            "SSR": "Server Side Rendering",
            "HMR": "Hot Module Replacement",
            "CMS": "Content Management System",
            "DDL": "Data Definition Language",
            "CSRF": "Cross Site Request Forgery",
            "USFM": "Unified Standard Format Markers",
            "DSS": "Decision Support System",
            "WCAG": "Web Content Accessibility Guidelines",
            "OWASP": "Open Web Application Security Project",
            "CGI": "Common Gateway Interface",
            "RFC": "Request For Comments",
            "BEM": "Block element modifier",
            "UPM": "unity package manager",
            "BDD": "Behavior-Driven Development",
            "GUID": "Globally Unique Identifier",
            "GPIO": "General-purpose input/output",
            "SOAP": "Simple Object Access Protocal",
            "RTF": "Rich Text Format",
            "AMP": "Accelerated Mobile Pages",
            "EBNF": "Extended Backus–Naur Form",
            "DSL": "domain-specific language",
            "WASM": "WebAssembly",
            "SEO": "Search Engine Optimization",
            "SFC": "System File Checker",
            "CSSOM": "CSS Object Model",
            "REST": "Representational State Transfer",
            "MERN": "Monogb/Express/React/Node",
            "MEAN": "Monogb/Express/Angular/Node",
            "MIME": "Multipurpose Internet Mail Extensions",
            "RPC": "Remote Procedure Call",
            "API": "Application Programming Interface",
            "JS": "JavaScript",
            "TS": "TypeScript",
            "FS": "FileSystem",
            "ES": "ECMAScript",
            "ES6": "ECMAScipt6",
            "ES7": "ECMAScipt7",
            "ES5": "ECMAScipt5",
            "ES4": "ECMAScipt4",
            "ES2016": "ECMAScipt2016",
            "ES2015": "ECMAScipt2015",
            "ES2017": "ECMAScipt2017",
            "ES3": "ECMAScipt3",
            "LTR": "Left To Right",
            "RTL": "Right To Left",
            "CI": "Continuous Integration",
            "CD": "Continuous Deployment"
        }
        self.abbr2detail = {key.lower(): value.lower() for key, value in abbrs.items()}
        self.detail2abbr = {value: key for key, value in self.abbr2detail.items()}
        pass
    
    def has_abbr(self, ne):
        if ne in self.abbr2detail:
            return True, self.abbr2detail[ne]
        for detail in self.detail2abbr:
            if fuzz.ratio(detail, ne) >= 85:
                return True, self.detail2abbr[detail]
        return False, None

class GraphBuilder(object):
    def __init__(self, save_gexf_file, ne2ne_file, ne2cate_file, filtered_keywords_file, filtered_topics_file, n_categories, load=False):
        self.save_gexf_file = save_gexf_file
        self.graph = nx.DiGraph()
        self.n_categories = n_categories
        self.neabbr = NEAbbr()
        self.npmname2gitname = {}
        if load:
            self.graph = nx.read_gexf(save_gexf_file)
        with open(ne2ne_file, "r", encoding="utf-8") as inf:
            self.ne2ne = json.load(inf)
        with open(ne2cate_file, "r", encoding="utf-8") as inf:
            self.ne2cate = json.load(inf)
        with open(filtered_keywords_file, "r", encoding="utf-8") as inf:
            self.filtered_keywords = json.load(inf)
        with open(filtered_topics_file, "r", encoding="utf-8") as inf:
            self.filtered_topics = json.load(inf)

    def __add_node(self, name, **kwargs):
        if not self.graph.has_node(name):
            self.graph.add_node(name, **kwargs)
    
    def __add_edge(self, src, dst, **kwargs):
        if not self.graph.has_edge(src, dst):
            self.graph.add_edge(src, dst, **kwargs)

    def extract_repo_info_from_github_url(self, github_url):
        first_idx = github_url.rfind("/")
        repo_name = github_url[first_idx+1:]
        second_idx = github_url[:first_idx].rfind("/")
        repo_owner = github_url[second_idx+1:first_idx]
        return repo_owner, repo_name
    
    def get_npmname2gitname(self, repo_stat_file):
        with open(repo_stat_file, "r", encoding="utf-8") as inf:
            for line in inf:
                obj = json.loads(line)
                github_url = obj.get("git_url", "")
                repo_owner, repo_name = self.extract_repo_info_from_github_url(github_url)
                if obj["name"] not in self.npmname2gitname:
                    self.npmname2gitname[obj["name"]] = repo_owner + "#" + repo_name
    
    def add_info_from_repo_stat(self, repo_stat_file):
        # 构建npm包名到Github包名的映射
        self.get_npmname2gitname(repo_stat_file)
        
        with open(repo_stat_file, "r", encoding="utf-8") as inf:
            for line in inf:
                obj = json.loads(line)

                npmname = obj["name"]
                repo_name = self.npmname2gitname[npmname]
                repo_owner = repo_name.split("#")[0]

                self.__add_node("$USER"+repo_owner, type="User")
                self.__add_node("$LIB"+repo_name, type="LibName")
                self.__add_edge("$LIB"+repo_name, "$USER"+repo_owner, kind="BelongToUser")
                
                for denp in obj.get("npm_dependencies", {}):
                    denp = self.npmname2gitname.get(denp, denp)
                    self.__add_node("$LIB"+denp, type="LibName")
                    self.__add_edge("$LIB"+repo_name, "$LIB"+denp, kind="HasDependence")
                for dev_denp in obj.get("npm_dev_dependencies", {}):
                    dev_denp = self.npmname2gitname.get(dev_denp, dev_denp)
                    self.__add_node("$LIB"+dev_denp, type="LibName")
                    self.__add_edge("$LIB"+repo_name, "$LIB"+dev_denp, kind="HasDevDependence")
                for topic in obj.get("git_topics", []):
                    topic = topic.lower()
                    if topic not in self.filtered_topics:
                        continue
                    self.__add_node("$TOPIC"+topic, type="GitHubTopic")
                    self.__add_edge("$LIB"+repo_name, "$TOPIC"+topic, kind="HasTopic")
                for keyword in obj.get("npm_keywords", []):
                    keyword = keyword.lower()
                    if keyword not in self.filtered_keywords:
                        continue
                    self.__add_node("$KEYWORD"+keyword, type="NPMKeyword")
                    self.__add_edge("$LIB"+repo_name, "$KEYWORD"+keyword, kind="HasKeyword")
    
    def add_ne_node(self, category, nes, lib_name):
        self.__add_node("$LIB"+lib_name, type="LibName")
        for ne in nes:
            category = self.ne2cate[ne]
            self.__add_node("$NE{}{}".format(category, ne), type="NE{}".format(category))
            has_abbr, abbr = self.neabbr.has_abbr(ne)
            if has_abbr:
                self.__add_node("$NE{}{}".format(category, abbr), type="NE{}".format(category))
                self.__add_edge("$NE{}{}".format(category, ne), "$NE{}{}".format(category, abbr), kind="SameToNE")
                self.__add_edge("$NE{}{}".format(category, abbr), "$NE{}{}".format(category, ne), kind="SameToNE")
            self.__add_edge("$LIB"+lib_name, "$NE{}{}".format(category, ne), kind="RelateToNE{}".format(category), weight=nes[ne])

    def add_info_from_ne(self, repo_stat_file, ne_file):
        with open(ne_file, "r", encoding="utf-8") as inf:
            for line in inf:
                splits = line.strip().split("\t")
                gitname = splits[0]
                for cate in range(self.n_categories):
                    nes = json.loads(splits[cate + 1])
                    self.add_ne_node(cate, nes, gitname)
    
    def export_gexf(self):
        nx.write_gexf(self.graph, self.save_gexf_file)
    
    def vis(self):
        # print(self.graph.edges)
        vis_graph = nx.DiGraph()
        vis_graph.add_node("bootstrap", type="LibName")
        for nbr in self.graph["bootstrap"]:
            if not vis_graph.has_node(nbr):
                vis_graph.add_node(nbr, type=self.graph.nodes[nbr]["type"])
            print(self.graph["bootstrap"][nbr])
            vis_graph.add_edge("bootstrap", nbr, type=self.graph["bootstrap"][nbr]["kind"])
            if self.graph.nodes[nbr]["type"] == "LibName":
                for nnbr in self.graph[nbr]:
                    if not vis_graph.has_node(nnbr):
                        vis_graph.add_node(nnbr, type=self.graph.nodes[nnbr]["type"])
                    vis_graph.add_edge(nbr, nnbr, type=self.graph[nbr][nnbr]["kind"])
        nx.write_gexf(vis_graph, "./vis.gexf")
        pass

if __name__ == "__main__":
    repo_stat_file = os.path.join(CLEANED_GH_REPOSITORY_FILE)
    ne_file = "./BeforeKG/refine_ne.txt"
    gb = GraphBuilder(
        save_gexf_file="./dls.gexf", 
        ne2ne_file="./BeforeKG/ne2ne.json", 
        ne2cate_file="./BeforeKG/ne2category.json",
        filtered_keywords_file="./BeforeKG/filtered_keywords.json", filtered_topics_file="./BeforeKG/filtered_topics.json", 
        n_categories=6, 
        load=False
    )
    gb.add_info_from_repo_stat(repo_stat_file)
    gb.add_info_from_ne(repo_stat_file, ne_file)
    gb.export_gexf()
