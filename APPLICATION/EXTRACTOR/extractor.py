#% pip install lxml
#% pip install beautifulsoup4

import pandas as pd
import re

class Extractor:
    def __init__(self):
        self.patterns = []
        
    def extract(self, text):
        matches = []

        for pattern in self.patterns: 
            matches.extend(re.findall(pattern, text))

        # 중복 제거 및 순서 유지
        unique_matches = list(dict.fromkeys(matches))

        return unique_matches
    
class PrecExtractor(Extractor):
    def __init__(self):
        self.patterns = [r'[가-힣]+ \d+\. \d+\. \d+\. 선고 [\d가-힣, ]+ 전원합의체 판결|[가-힣]+ \d+\. \d+\. \d+\. 선고 [\d가-힣, ]+ 판결 참조',
        r'[가-힣]+ \d+\. \d+\. \d+\. 선고 [\d가-힣, ]+ 전원합의체 판결|[가-힣]+ \d+\. \d+\. \d+\. 선고 [\d가-힣, ]+ 판결 등 참조',
        r'[가-힣]+ \d+\. \d+\. \d+\. 선고 [\d가-힣, ]+ 전원합의체 판결|[가-힣]+ \d+\. \d+\. \d+\. 선고 [\d가-힣, ]+ 판결 등 참조',
        r"대법원 \d{4}\. \d{1,2}\. \d{1,2}\. 선고 \d{2,5}다\d{1,5} 판결(?:, 대법원 \d{4}\. \d{1,2}\. \d{1,2}\. 선고 \d{2,5}다\d{1,5} 판결)* 참조",
        r"대법원 \d{4}\. \d{1,2}\. \d{1,2}\.[가-힣] \d{2,5}[가-힣]\d{1,5} 결정(?:, 대법원 \d{4}\. \d{1,2}\. \d{1,2}\. 선고 \d{2,5}[가-힣]\d{1,5})* 판결 등 참조",
        r"\( 대법원 \d{4}\. \d{1,2}\. \d{1,2}\. 선고 \d{2,5}[가-힣]\d{1,5}(?:, 대법원 \d{4}\. \d{1,2}\. \d{1,2}\. 선고 \d{2,5}[가-힣]\d{1,5})* 판결 등 참조 \)",
        r"\(대법원 \d{4}\. \d{1,2}\. \d{1,2}\.자? \d{1,5}[가-힣]\d{1,6} 결정(?:, 대법원 \d{4}\. \d{1,2}\. \d{1,2}\. 선고 \d{1,5}[가-힣]\d{1,6} 판결)* 등 참조\)",
        r'\( 대법원.*?취지 참조\)',
        r'\(대법원.*?취지 참조\)',
        r'\( 대법원.*?등의 취지 참조\)',
        r'\(대법원.*?등의 취지도 참조\)',
        r'\( 대법원.*?등의 취지도 참조\)',
        r"대법원 \d{4}\. \d{1,2}\. \d{1,2}\. 선고 \d{4,5}[가-힣]\d{1,5} 판결",
        r"대법원 \d{4}\. \d{1,2}\. \d{1,2}\. 선고 \d{4,5}[가-힣]\d{1,5}판결 참조",
        r'\(대법원.*?취지 등 참조\)']
        
class LawExtractor(Extractor):
    import pandas as pd
    # 일반화 개선점 존재
    # groups 전체를 포함시키는 것 고려할 필요
    # 현재는 1, 2번 group에 대해서만 적용됨
    # API를 사용하기 때문에 권한 있는 user oc를 user에 입력 
    def __init__(self, user_oc):
        self.user = user_oc
        self.law_df = pd.read_csv("EXTRACTOR/lawNameCrawl.csv")
        self.patterns = [
            r'^제*\d+조[^의]',
            r'^제*\d+조[^의]제*\d+항',
            r'^제*\d+조[^의]제*\d+호',
            r'^제*\d+조[^의]제*[가-힣]목',
            r'^제*\d+조[^의]제*\d+항제*\d+호',
            r'^제*\d+조[^의]제*\d+항제*[가-힣]목',
            r'^제*\d+조[^의]제*\d+호제*[가-힣]목',
            r'^제*\d+조[^의]제*\d+항제*\d+호제*[가-힣]목',
            r'^제*\d+조의\d+',
            r'^제*\d+조의\d+제*\d+항',
            r'^제*\d+조의\d+제*\d+호',
            r'^제*\d+조의\d+제*[가-힣]목',
            r'^제*\d+조의\d+제*\d+항제*\d+호',
            r'^제*\d+조의\d+제*\d+항제*[가-힣]목',
            r'^제*\d+조의\d+제*\d+호제*[가-힣]목',
            r'^제*\d+조의\d+제*\d+항제*\d+호제*[가-힣]목'
        ]
        
    @staticmethod
    def no_space(str):
        return str.replace(" ", "")
    
    @staticmethod
    def urlBase(user) : 
        base_url = f'http://www.law.go.kr/DRF/lawService.do?OC={user}&type=XML'
        return base_url
    
    @staticmethod
    def urlLaw(base_url, lawNameDf, lawNameDfIdx):
        law_id = lawNameDf['법령ID'][lawNameDfIdx] #법령 ID부분
        return  base_url + '&ID=' + f'{law_id}'
    
    @staticmethod
    def getJo(string, url):
        string = string.rstrip('의')

        if '조의' in string:
            match = re.search(r'(\d+)조의(\d+)', string)
            if match:
                p_front, p_behind = int(match.groups()[0]), int(match.groups()[1])
                url += f"&JO={p_front:04d}{p_behind:02d}"
        else:
            match = re.search(r'(\d+)조', string)
            if match:
                p = int(match.groups()[0])
                url += f"&JO={p:04d}00"
        return url
    
    @staticmethod
    def getHo(string, url):
        if '호의' in string:
            match = re.search(r'(\d+)호의(\d+)', string)
            if match:
                p_front, p_behind = int(match.groups()[0]), int(match.groups()[1])
                url += f"&HO={p_front:04d}{p_behind:02d}"
        else:
            match = re.search(r'(\d+)호', string)
            if match:
                p = int(match.groups()[0])
                url += f"&HO={p:04d}00"
        return url
    
    @staticmethod
    def getMok(string, url):
        match = re.search(r'([가-힣])목', string)
        if match: url += f'&MOK={match.groups()[0]}'
        return url
    
    @staticmethod
    def getHang(string, url):
        match = re.search(r'(\d+)항', string)
        if match:
            hang = int(match.groups()[0])
            url += '&HANG=' + f'{hang:04d}00'
        return url
    
    def get_sub_law_name(self, text):
        search_results = []
        compiled_patterns = [re.compile(pattern) for pattern in self.patterns]

        for i, pattern in enumerate(compiled_patterns):
            result = pattern.search(text)
            search_results.append(result)

        return tuple(search_results)

    def argmax(self, search_results:tuple): 
        # 찾은 패턴 중 가장 긴 case의 텍스트(group)와 그 인덱스를 반환
        search_list = []
        search_list = [len(res.group()) if res is not None else 0 for res in search_results]
        argmax = search_list.index(max(search_list))

        return search_results[argmax], argmax     

    def extract_law_info(self, prec_text):
        law_df = self.law_df
        user = self.user

        body_text = prec_text
        law_names = self.law_df['법령명한글']

        law_detail = "" # sub_law_name
        name_flag = 0 # 조항호목 없는 경우 100, 구법인 경우 -1                

        urls = []
        return_law_name_list = []

        # 목차에서 검색하지 말고 나중에 API로 호출할 수 있게 수정 필요
        for i, law_name in enumerate(law_names):
            no_space_body = self.no_space(body_text)
            no_space_law_name = self.no_space(law_name)


            law_name_start = no_space_body.find(no_space_law_name)

            # 법령명 시작점이 존재하지 않는 경우 다음 루프로 이동
            if law_name_start == -1:
                continue

            search_results = re.finditer(no_space_law_name, no_space_body)
            

            for law_name_loc in search_results:
                # print(no_space_body[law_name_loc.end():])

                # 법령명 이후 조,항,목 출현 확인
                body_after_name = no_space_body[law_name_loc.end():]

                def bracket(text):
                    # 괄호를 찾아서 반환
                    return re.search(r'^\(.+\)', text)
                def is_old():
                    # 구법 여부 확인: 찾은 법령명 앞에 "구"가 있는지 체크
                    # 괄호가 없어야 함
                    if no_space_body[law_name_loc.start()-1]=="구":
                        return True
                    else:
                        return False

                search_bracket = bracket(body_after_name)

                if search_bracket:
                    body_after_name = body_after_name[bracket(body_after_name).end() : ] #괄호 이후를 slice          


                # 조, 항, 목 찾아서 제일 긴 이름으로 반환
                # 예) 부동산 등기법 제3조 제1항, 부동산 등기법 제3조 -> 부동산 등기법 제3조 제1항 반환
                res = self.get_sub_law_name(body_after_name)
                finded_long, finded_long_index = self.argmax(res)

                # 법령 검색
                url = self.urlBase(user)

                # 조항호목이 없는 경우
                if finded_long is None : 
                    # name_flag는 구법일 경우 -1, 조항호목이 없는 경우 100 
                    name_flag = 100
                    url = self.urlLaw(url, law_df, i) 
                else:
                    # 그 외에는 finded_long_index(조항호목 정보)
                    name_flag = finded_long_index
                    law_detail = finded_long
                    url = self.urlLaw(url, law_df, i) 
                    url += '&target=lawjosub' # 법령 조항호목 검색 // josub 없이 검색 가능
                    url = self.getHang(finded_long.group(), url) # 항
                    url = self.getJo(finded_long.group(), url)   # 조
                    url = self.getHo(finded_long.group(), url)   # 호

                # 구법 여부 확인
                # search_bracke logic 확인 필요
                if is_old() and search_bracket:
                    name_flag = -1 # 구법 flag

                    if finded_long is None:
                        law_detail = search_bracket.group()
                    else:
                        # finded_long이 None이 아닐 경우
                        law_detail = search_bracket.group()+finded_long.group()

                elif search_bracket:
                    # 괄호만 있으나 구법인 경우
                    if "개정되기전의것" in search_bracket.group():
                        name_flag = -1
                        law_detail = search_bracket.group()+finded_long.group()
                        # 법령상세링크 대신 조합해서 사용하게 수정
                    elif finded_long:
                        name_flag = finded_long_index
                        law_detail = finded_long

                else:
                    # 패턴 중 "목"이 있는 경우 url 목 추가
                    pattern_idx_mok = [3,5,6,7,11,13,14,15] # patterns에서 "목"이 포함된 위치
                    if finded_long_index in pattern_idx_mok:
                        url = self.getMok(finded_long.group(), url) # 목

                try:
                    full_law_name = law_name+' '+law_detail.group()
                except:
                    full_law_name = law_name

                return_law_name_list.append(full_law_name.strip())
                urls.append(url)

        # 후 처리
        return_law_name_list = [ x if x[-1] in ['조','항','호','목','법','칙','률'] else x[:-1] for x in return_law_name_list ]

        return urls, return_law_name_list
    
    @staticmethod
    def return_text(url):
        # API로 텍스트 반환
        txt = []
        from bs4 import BeautifulSoup
        import requests 

        if len(url.split("&target=lawjosub")) == 1:
            url += "&target=lawjosub&JO=000000"

        response = requests.get(url)
        content = response.text

        # BeautifulSoup 객체 생성
        soup = BeautifulSoup(content, 'xml')

        # 태그 이름 리스트
        tags = ["법령명_한글", "조문내용", "항내용", "호내용", "목내용"]

        # 모든 태그의 텍스트 추출
        for tag in tags:
            elements = soup.find_all(tag)
            for element in elements:
                # print(f"{element.get_text().strip()}")
                txt.append(element.get_text().strip())

        return txt
    
    def text_post_processing(self, urls, law_names):         
        from collections import defaultdict
        temp = defaultdict(list)
        infos = defaultdict(list)
        
        for url, law_name in zip(urls, law_names):
            text = ' '.join(self.return_text(url))
            temp[law_name].append((url, text))

        for law_name, url_text in temp.items():
            # 각 항목의 텍스트 길이 계산
            lengths = [len(item[1]) for item in url_text]

            # 가장 긴 텍스트를 가진 항목 찾기
            max_length_index = lengths.index(max(lengths))

            # 가장 긴 텍스트를 저장
            info = url_text[max_length_index]
            infos[law_name] = info
        return infos
    
if __name__ == "__main__": 
    # user 입력
    user = 'sillontea'
    df = pd.read_csv("civil.csv")
    prec = df.iloc[3]
    
    law_extractor = LawExtractor(user)
    prec_extractor = PrecExtractor()

    ref_prec = prec_extractor.extract(prec['전문'])
    urls, ref_law = law_extractor.extract_law_info(prec['전문'])
    
    # 다음은 main에서 작동할 로직 일부
    # data에 원래 있던 판례 및 조문
    # list(itertools.chain(ast.literal_eval(prec['참조조문'])))

    for url, law_name in zip(urls, law_names):
        url = ''.join(url.split('type=XML&'))+"&type=HTML"
        print(law_name, url)

    print()

    if ref_prec:
        print(ref_prec)
    else:
        print("참조판례가 존재하지 않습니다.\n")
        print("======"*5)
    
    for url in urls:
        print('\n'.join(law_extractor.return_text(url)), end='\n\n')