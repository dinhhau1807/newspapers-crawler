import scrapy
import os
import re
import urllib
from urllib.parse import urlparse
from urllib.request import urlopen
from bs4 import BeautifulSoup
import tldextract
import newspaper


class SpiderManSpider(scrapy.Spider):
    name = 'spiderman'
    handle_httpstatus_list = [404, 500, 502, 503]
    sub_folder_path = ''

    TOPICS = ['arts',
              'books', 'business',
              'daily',
              'education', 'economy', 'entertainment', 'environment',
              'fashion', 'food',
              'health',
              'international',
              'lifestyle', 'life', 'living',
              'money',
              'nation', 'news', 'new',
              'opinion',
              'politics',
              'sciencetech', 'science', 'society', 'sports', 'sport', 'style',
              'technology', 'tech', 'travel',
              'videos', 'video',
              'weather', 'world']

    ACCEPT_EXTENSION = ['.html', '.htm']

    #####################################################################################
    # Kiểm tra xử lý URL
    def is_valid_url(self, url):
        regex = re.compile(
            r'^(?:http|ftp)s?://'  # http:// or https://
            # domain...
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        result = re.match(regex, url) is not None

        if result:
            try:
                url_parsed = urlparse(url)
                url = url_parsed.scheme + '://' + url_parsed.netloc

                getUrl = urllib.request.urlopen(url)
            except urllib.error.HTTPError as e:
                # Return code error (e.g. 404, 501, ...)
                # ...
                print('HTTPError: {}'.format(e.code))
                result = False
            except urllib.error.URLError as e:
                # Not an HTTP-specific error (e.g. connection refused)
                # ...
                print('URLError: {}'.format(e.reason))
                result = False
            else:
                # 200
                # ...
                result = True

        return result

    #####################################################################################
    # Xử lý tạo folder
    def create_root_folder(self, url):
        uri = tldextract.extract(url)
        folder_name = 'NEWSPAPERS/' + uri.domain.capitalize()
        if not os.path.isdir('NEWSPAPERS'):
            os.mkdir('NEWSPAPERS')
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)
        return folder_name

    def create_sub_folder(self, root_folder_name, sub_folder_name):
        path = os.path.join(root_folder_name, sub_folder_name)
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    #####################################################################################
    # Xử lý lấy topics
    def parse_nav_links(self, nav_links):
        result = {}
        for link in nav_links:
            href = str(link.get('href'))
            for topic in self.TOPICS:
                if topic in href and topic in ['nation', 'news', 'new']:
                    pattern = "\b{0}\b".format(topic)
                    pattern = re.compile(pattern)
                    if pattern.match(href):
                        result[topic] = href
                    break
                if topic in href:
                    result[topic] = href
                    break
        return result

    def get_topics_advanced(self, url):
        html_page = urlopen(url)
        soup = BeautifulSoup(html_page, "html.parser")
        result = {}

        nav_links = soup.select('nav a')
        # print(nav_links)
        result = self.parse_nav_links(nav_links)

        if len(result) == 0:
            # print('????? nav_links time 0')
            nav_links = soup.select('[class*="nav"] a')
            # print(nav_links)
            result = self.parse_nav_links(nav_links)

        if len(result) == 0:
            # print('????? nav_links time 1')
            nav_links = soup.select('[data-testid*="nav"] a')
            # print(nav_links)
            result = self.parse_nav_links(nav_links)

        if len(result) == 0:
            # print('????? nav_links time 2')
            nav_links = soup.select('[class*="topics"] a')
            # print(nav_links)
            result = self.parse_nav_links(nav_links)

        for key, value in result.items():
            if value[0] == '/':
                result[key] = url + value

        return result

    def get_topics(self, url):
        # Build newspaper
        home_page = newspaper.build(url)
        links = home_page.category_urls()
        # print('>>>> newspapers')
        # print(links)

        result = {}

        # print('\n>>>> parts[i]')
        for link in links:
            url_link = urlparse(link)
            parts = url_link.path.split('/')
            if any(ext in parts[-1] for ext in self.ACCEPT_EXTENSION):
                parts[-1] = parts[-1].split('.')[0]
            for i in range(1, len(parts)):
                # print(parts[i])
                if parts[i] and parts[i].lower() != 'index' and any(topic in parts[i] for topic in self.TOPICS):
                    # print('^ ' + parts[i].upper())
                    result[parts[i]] = link

        if len(result) == 0:
            # print('>>>> joined advanced')
            result = self.get_topics_advanced(url)

        return result

    #####################################################################################
    # Xử lý hiển thị topics
    def create_topic_name(self, topic):
        parts = topic.split('-')
        topic = ' '.join([part.capitalize() for part in parts])
        return topic

    def print_topics(self, topics):
        for topic in topics:
            print('{0}. {1}'.format(topics.index(topic) + 1,
                                    self.create_topic_name(topic)))

    #####################################################################################
    # Lấy lựa chọn người dùng
    def get_choice(self, topics):
        choice = -1
        while choice not in range(1, len(topics) + 1):
            try:
                choice = int(input('Hay chon chu de co trong danh sach: '))
            except ValueError:
                print('!!! Hay nhap vao gia tri so nguyen.')
                choice = -1
        return choice

    #####################################################################################
    # Xử lý lấy articles
    def filter_article_links(self, url, topic, links, article_links, type=0):
        for link in links:
            if link:
                if link[0] == '/':
                    link = url+link

                # Check link has accept extension
                parts = link.split('/')
                if len(parts) > 1:
                    last_part = parts[-1]
                    split_ext = last_part.split('.')
                    if len(split_ext) > 1 and split_ext[-1] != '' and '.'+split_ext[-1] not in self.ACCEPT_EXTENSION:
                        continue

                if type == 0:
                    # Pattern match YYYY-MM-DD, YYYY/MM/DD
                    pattern = "([12]\d{3}(-|\/)(0[1-9]|1[0-2])(-|\/)(0[1-9]|[12]\d|3[01]))"
                    pattern = re.compile(pattern)
                    if pattern.search(link) and topic in link:
                        article_links.append(link)

                if type == 1:
                    # Pattern match YYYY-MM-DD, YYYY/MM/DD
                    pattern = "([12]\d{3}(-|\/)(0[1-9]|1[0-2])(-|\/)(0[1-9]|[12]\d|3[01]))"
                    pattern = re.compile(pattern)
                    if pattern.search(link):
                        article_links.append(link)

                if type == 2:
                    if topic in link and any(ext in link for ext in self.ACCEPT_EXTENSION):
                        article_links.append(link)

    def get_article_links(self, url, topic, topic_url):
        page = urlopen(topic_url)
        soup = BeautifulSoup(page, "html.parser")
        links = [link.get('href') for link in soup.select('a')]

        # print(links)

        article_links = []
        # print('-------> filter root')
        self.filter_article_links(url, topic, links, article_links, 0)

        if len(set(article_links)) <= 3:
            # print('-------> filter level 1')
            self.filter_article_links(url, topic, links, article_links, 1)

        if len(set(article_links)) <= 3:
            # print('-------> filter level 2')
            self.filter_article_links(url, topic, links, article_links, 2)

        return list(set(article_links))

    #####################################################################################
    # Hàm start
    def start_requests(self):
        # Kiêm tra url hợp lệ
        url = input('Nhap URL: ')
        while self.is_valid_url(url) is False:
            print('Url nhap vao khong hop le hoac khong duoc phep crawl!')
            url = input('Nhap lai URL: ')

        # Xoá / ở cuối url nếu có
        urlparsed = urlparse(url)
        url = urlparsed.scheme + '://' + urlparsed.netloc
        print('Lay chu de tu lien ket: ' + url)
        print('-------------------------------------')

        # Tạo folder root
        root_folder_name = self.create_root_folder(url)

        # Lấy topics
        topics = self.get_topics(url)

        # Hiển thị topics
        topic_keys = list(topics.keys())
        self.print_topics(topic_keys)

        # Lấy lựa chọn topic của người dùng
        choice = self.get_choice(topics)

        # Lấy topic url
        topic_url = topics[topic_keys[choice-1]]
        print('>>> Crawling chu de tu lien ket: {0}'.format(topic_url))

        # Tạo sub folder
        self.sub_folder_path = self.create_sub_folder(
            root_folder_name, self.create_topic_name(topic_keys[choice-1]))

        # Lấy links các bài viết
        article_links = self.get_article_links(
            url, topic_keys[choice-1], topic_url)
        print('Tim thay ({}) bai viet...'.format(len(article_links)))

        # Crawl hết nội dung theo chủ đế
        for link in article_links:
            yield scrapy.Request(url=link, callback=self.parse, errback=self.error_function)

    def parse(self, response):
        parts = urlparse(response.url).path.split('/')
        parts[-1] = parts[-1].split('.')[0]
        if parts[0] == '':
            parts.remove(parts[0])

        filename = '-'.join(parts) + '.html'

        with open(self.sub_folder_path+"/"+filename, 'wb') as f:
            f.write(response.body)

        print('Saved file %s' % filename)
        self.log('Saved file %s' % filename)

    def error_function(self, failure):
        self.logger.error(repr(failure))
        print('ERROR!!!')
