from bs4 import BeautifulSoup
import urllib.request
import json
import os
class Data:
    def __init__(self, title, description, paras) -> None:
        self.title = title
        self.description = description
        self.paras = paras

# url = 'https://cn.chinadaily.com.cn/a/202302/21/WS63f42ccca3102ada8b22fe7b.html'

def crawl_News(url):
    data = {"content": []}
    page = urllib.request.urlopen(url)
    soup = BeautifulSoup(page, 'html.parser')
    if "www.theguardian.com" in url:
        # Get title
        title = soup.select('.dcr-y70mar')[0].text.strip()
        # Get description
        description = soup.select('div.dcr-iuxtqj')[0].text.strip()
        # Get content
        content = soup.find('div', class_="article-body-commercial-selector")
        p_tags = content.find_all("p")
        paras = []
        for p in p_tags:
            para = p.text
            paras.append(para)
        content = Data(title, description, paras)
        data["content"].append({"title": content.title, "description": content.description, "paras": content.paras})
        title = data['content'][0]['title']
        description = data['content'][0]['description']
        paras = data['content'][0]['paras']
        return title, description,paras
    elif "dantri.com.vn" in url:
        data = {"content": []}
        page = urllib.request.urlopen(url)
        soup = BeautifulSoup(page, 'html.parser')
        # Get title
        title = soup.select('h1.title-page')[0].text.strip()
        # Get description
        description = soup.select('h2.singular-sapo')[0].text.strip()
        # Get content
        content = soup.find('div', class_="singular-content")
        p_tags = content.find_all("p")
        paras = []
        for p in p_tags:
            para = p.text
            if not ('.' == para[-1] and 'Ảnh' in para):
                paras.append(para)
            else:
                continue
        content = Data(title, description, paras)
        data["content"].append({"title": content.title, "description": content.description, "paras": content.paras})
        title = data['content'][0]['title']
        description = data['content'][0]['description'].replace('(Dân trí) - ','')
        paras = data['content'][0]['paras']
        return title, description,paras
    elif "chinadaily.com.cn" in url:
        # Get title
        title = soup.select('h1.dabiaoti')[0].text.strip()
        description = title
        # Get content
        content = soup.find('div', class_="article")
        p_tags = content.find_all("p")
        paras = []
        for p in p_tags:
            para = p.text.replace('。','. ')
            para = para.replace('？','?')
            paras.append(para)
        # Store the data in a Data object
        content = Data(title, description, paras)
        data["content"].append({"title": content.title, "description": content.description, "paras": content.paras})
        title = data['content'][0]['title']
        description = data['content'][0]['title']
        paras = data['content'][0]['paras']
        return title, description,paras
    elif "msk.kp.ru"  in url:
        # Get title
        title = soup.select('.eyeguj')[0].text.strip()
        # Get description
        description = soup.select('.nFVxV')[0].text.strip()
        # Get content
        p_tags = soup.find_all("p", {"class": "dqbiXu"})
        paras = []
        for p in p_tags:
            para = p.text
            paras.append(para)
        content = Data(title, description, paras)
        data["content"].append({"title": content.title, "description": content.description, "paras": content.paras})
        title = data['content'][0]['title']
        description = data['content'][0]['description']
        paras = data['content'][0]['paras']
        return title, description,paras
    else:
        print("can't work with this page")

# crawl = crawl_News('https://dantri.com.vn/the-gioi/ukraine-khoa-mui-tien-cong-cua-nga-o-bien-den-truoc-tran-danh-lon-20230213114746124.htm')
# print(crawl)
