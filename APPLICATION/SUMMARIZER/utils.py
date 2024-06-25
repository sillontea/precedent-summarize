# 2024.6.21 SectionSplitter class update:
#    add - level, is_fact_head, no_change
#    modified - init, _head_check, group_head_content
# 다시 쓰는 판결이유는 전부 하나의 그룹으로 처리( 관련 파트 group_head_content )
# 2024.6.22. SectionSplitter 오류 수정

import re
import pandas as pd
    
class SentenceSplitter:
    def __init__(self):
        self.get_patterns() # pattern 변수 설정
        self.pattern = re.compile(self.check_patterns, re.VERBOSE|re.IGNORECASE)

    def split_sent(self, text:str)->list:
        """
        Splits the given text into sentences while handling exceptions such as quotes and specific patterns.

        Args:
            text(str): The text to be split into sentences.
        Returns:
            list: A list of sentences.
        """
        
        pattern = self.pattern

        # 문장부호에 특수문자가 붙거나 유니코드일 경우 처리
        text = self.cleansing(text)
        
        # 예외처리: 문장 끝에 .(공백없음)으로 끝나거나 .이 없는 경우
        sample = self.new_sample(text)
        splited_list = []
        prev_end = 0
        in_quote = False
        
        
        # '. '으로 끝나는 모든 경우를 find
        for sent in re.finditer('(.\. ?)', sample): # |(\? ?) 제외
            start = sent.span()[0]
            end = sent.span()[1]

            token = sample[max(start-8, 0):end-1]

            # 인용문 예외처리
            # 인용문 1. 인용 부호
            # 인용 부호가 나타나면 인용문 시작 in_quote = True
            # 인용 부호가 다시 나타나면 인용문 끝 in_quote = False
            if re.search(self.quote_pattern, sample[prev_end:end]):
                in_quote = not in_quote
            # 인용문 2. 본지 인용(~항)
            # 가.항을 볼 때, 1)항에 따르면 등의 경우
            not_ref = re.search("([가-힣1-9][\.\)]\s?항)", sample[start:end+3]) is None
                
            # '. ' 앞뒤(토큰)에 기본 예외 패턴이 없고, 인용문이 아닐 경우 문장으로 나누기
            if (pattern.search(token) == None) and not in_quote and not_ref:
                # 날짜정보, 토지 단위 등이 나타난 경우, 인용기호 등 추가 예외 처리
                if self.is_not_exception(sample, token, start, end):
                    splited_list.append(sample[prev_end:end].strip())
                    prev_end = end
                
            elif pattern.search(token) != None:
                # 마지막일 경우
                if len(sample) == end:
                    splited_list.append(sample[prev_end:end].strip())
                    
        if len(splited_list) == 0:
            splited_list.append(sample.strip())
        return splited_list


    # '.●'와 같은 특이 케이스 처리를 위해 추가(10.5)
    # '６８.'특수문자 처리 안 함
    # \n, \t, \xa0 관련 제거(1.10)
    # space 가 많이 들어가 있을 경우 1개로 줄이기
    def cleansing(self, text):
        text = text
        cleansed_text = ''
        
        # This line gets rid of withe spaces more than one to only one space.
        text = re.sub(" +", " ", text)
        
        try:
        # 유니코드 문자 제거
            p = re.compile(self.unicode_pattern, re.VERBOSE)
            cleansed_text = p.sub(" ", text)
            cleansed_text = cleansed_text.strip()
        except:
            pass
        
        try:
            # 특수기호 앞 공백 추가
            p = re.compile(self.symbol_pattern, re.VERBOSE)
            match_list = [ m.group() for m in p.finditer(cleansed_text)]
            match_list= list(set(match_list))

            for match in match_list:
                h = Substitutable(cleansed_text)
                cleansed_text = h.sub(match, ' '+match)
        except:
            pass
        
        try:
            # 마침표 특수문자 case 처리
            text = re.sub(r"．", ".", text)
        except:
            pass

        return cleansed_text
    
    def _bracket(self, sent):
        splitted = []
        pattern = re.compile("[【\[].*?[】\]]") # "?"non-greedy match: 최소 문자열

        while sent:
            match = pattern.search(sent)
            if match:
                start, end = match.span()
                if start > 0:
                    splitted.append(sent[:start].strip())
                splitted.append(sent[start:end].strip())
                sent = sent[end:].strip()
            else:
                splitted.append(sent.strip())
                break

        return splitted
    
    def new_sample(self, sample):
        try:
            if sample.endswith('. '):
                return sample
            elif sample.endswith('.'):
                return sample + ' '
            else:
                return sample
        except:
            return sample

    def is_not_exception(self, sample, token, start, end):
        # 날짜정보 예외처리 0000.00.00.
        date_check_1 = re.search(self.date_pattern_1, token)
        date_check_2 = (re.search(self.date_pattern_21, sample[start-8:end+3])) or (re.search(self.date_pattern_22,sample[start-8:end]))
        date_check_3 = re.search(self.date_pattern_3, sample[start-12:end])
        date_check = (date_check_1 is None) and (date_check_2 is None) and (date_check_3 is None)
        
        # 기타 패턴 [0-9]+㎡ 등 예외 처리
        etc_check = re.search(self.etc_pattern, sample[start-8:end+4]) is None
        
        # 인용문 패턴 예외 처리
        quote_check = re.search(self.quote_pattern, sample[start:end+4]) is None
        
        return date_check and etc_check and quote_check

    def get_patterns(self):
        self.check_patterns = """ (eqs?\.\s?)|(figs?\.\s?)|(sec\.\s?)|(et\.\set\.\s?)|(et\.?\sal\.?\s?)|(e\.g\.\s?)|(i\.e\.\s?)|(no\.\s?)\
          |\s?([0-9]{1}\.?\s?)$|(.?[0-9]+\.\s?)$|(vs\.\s?)|(ref\.\s?)|(cf\.\s?)\ # 논문 관련 약자들
          |(\s?\[\s?\w+\s?\w*\\s?]) | (\<\w+\>) | (【\w+=?\w*】\s?)\             # | ([\(\)])\ 괄호문자
          |.?(www\.?\s?)|(\.or\.?)|(\.go\.?)|(\.co\.?)|([0-9a-z]+@[a-z]+\.?)\   # 링크주소1 www. segye. co,
          |(\([a-z]+\.?)\      # (abc. 은 문장의 끝이 아닐 것
          |\s?([A-Z]{1}\.?\s?)$|^\s?([A-Z]{1}\.)\ # 이름, 등
          |.*(\s+[가-힣]\.\s?)\
          |^(\s*[가-힣]\.\s?)\
          |(\s?[ⅠⅠⅡⅢⅣⅤⅩⅢⅧⅳⅦⅲⅪⅡⅣⅴⅱⅨⅰⅠⅥⅩⅤⅡ]+\.?\s?)\
          |.*(\.\.\.).*\
          """
        
        self.date_pattern_1 = '([0-9]{4}^(\]\))\.?)'
        self.date_pattern_21 = '.?\s?([0-9]{4}\.\s?[0-9]+\.\s?[0-9]+\.?)'
        self.date_pattern_22 = '.?\s?([0-9]{4}\.\s?[0-9]+\.?)'
        self.date_pattern_3 = '.?([0-9]{4}\.\s?[0-9]+\.\s?[0-9]+\.?)'
        
        self.unicode_pattern = """(\\uf076)|(\\xa0)|(\\x03)|(\\uf0a7)|(\\ufeff)|(\\u200c)|(\\n)|(\\t)"""
        self.symbol_pattern = """[■▦◆▲●▶▽△◇©ⓒ①②③④⑤⑥⑦®㈜※【】]  #특수문자"""
        self.etc_pattern = "[0-9]+\.\s?[0-9]+㎡?"
        self.quote_pattern = '\.\s?[\[\]"”““‟”’’❝❞’〝〞\"\'‘‛❛❜\)].*'
      
# ref. https://stackoverflow.com/questions/15175142/how-can-i-do-multiple-substitutions-using-regex
class Substitutable(str):
    def __new__(cls, *args, **kwargs):
        newobj = str.__new__(cls, *args, **kwargs)
        newobj.sub = lambda fro,to: Substitutable(re.sub(fro, to, newobj))
        return newobj
    
class SectionSplitter:
    
    def __init__(self, precedent):
        self.split_sent = SentenceSplitter().split_sent
        self._bracket = SentenceSplitter()._bracket
        self.raw_data = precedent
        
        try:
            self.data = precedent["전문"].split('\n')
        except:
            self.data = precedent.split('\n')
        
        # 2024.6.22. Changed the processign order
        self.data = self.split_content_sent(self.data)
        grouped_data = self.head_check(self.data)
        self.grouped_data = self.group_head_content(grouped_data)
    
    
    # Level 판별 함수
    def level(self, sent):
        level_patterns = [
            "^([1-9]+\.).*[^.,\s]$",  # Level 1
            "^([가-하]+\.).*[^.,\s]$",  # Level 2
            "^([1-9]+\)).*[^.,\s]$",  # Level 3
            "^([가-하]+\)).*[^.,\s]$",  # Level 4
        ]
        for i, pattern in enumerate(level_patterns, 1):
            if re.search(pattern, sent):
                return i
        return 4
    
    # 2024.6.22. A logical error is modified 
    def no_change(self, blv, bfct, lv):
        if lv == 1:
            return False
        elif bfct and lv > blv:
            return True  # no_change
        return False

    # 사실 HEAD 여부 판별 함수
    def is_fact_head(self, sent):
        pattern = "사실관계|범죄사실|인정사실|사실의인정|판단(의\s*요지)?$|공소사실"
        return bool(re.search(pattern, sent))
    
    # HEAD 체크 함수
    def _head_check1(self, sent):
        # 괄호 헤드 체크
        regex = re.compile("^【.+】$")
        return True if re.match(regex, sent) else False

    def _head_check2(self, sent):
        # 글머리 헤드 체크
        # 문장의 끝 지점에는 문장부호가 존재하지 않음
        def _level1(sent):
            # 1.제목 또는 가.제목
            regex = "^([1-9]+\.|[가-하]+\.).*[^.,\s]$"
            return bool(re.search(regex, sent))

        def _level2(sent):
            # 1)제목 또는 가)제목
            regex = "^([1-9]+\)|[가-하]+\)).*[^.,\s]$"
            return bool(re.search(regex, sent))

        return _level1(sent) or _level2(sent)


    def head_check(self, sentences:list):
        after_iyu = False
        _head_check1, _head_check2 = self._head_check1, self._head_check2
        
        new_sentences = []
        row = None

        for sent in sentences:
            # if sent[1:-1].replace(' ', '') == "이유":
            #    after_iyu = True

            if re.search("(이유)", sent.replace(" ", "")):
                after_iyu = True

            if after_iyu:
                if _head_check1(sent):
                    row = ("HEAD", sent)
                    # print("HEAD", sent)

                elif _head_check2(sent):
                    row = ("HEAD", sent)
                    # print("HEAD", sent)

                else:
                    row = ("CONTENT", sent)

            else:
                row = ("NO", sent)
                # print("No", sent)
            new_sentences.append(row)
            row = None
        return new_sentences

    # 2024.6.21. group 분할 로직 추가
    def group_head_content(self, data):
        result = []
        group_id = 0
        current_group = []

        blv = 1
        bfct = False
        _rewrite = False

        for label, text in data:
            if label == 'NO':
                continue
                
            if _rewrite:
                current_group.append((label, text))
                continue

            if label == 'HEAD':
                # 다시쓰는 판결이유 예외처리
                _rewrite = bool(re.search("다시쓰는판결이유", text.replace(" ", "")))
    
                lv = self.level(text)            
                is_fact = self.is_fact_head(text)
                
                # 사실이 포함된 목차인 경우
                if is_fact and "원심" not in text:
                    bfct = True
                    blv = lv
                    
                # 현재 목차가 사실 목차 보다 상위 목차인 경우(not no_change)
                # 새로운 그룹 아이디 부여
                if _rewrite or not self.no_change(blv, bfct, lv): 
                    # 다만 이전 데이터가 텍스트(CONTENT)가 아닌 목차(HEAD)정보일 경우 그룹 아이디를 변경하지 않음    
                    if current_group and current_group[-1][0] == 'CONTENT':
                        # 새로운 HEAD가 시작될 때, 현재까지 저장된 정보를 결과에 저장
                        result.extend([(group_id, l, t) for l, t in current_group])
                        group_id += 1 # 새로운 그룹 ID 시작
                        current_group = []
                    # HEAD 정보를 새롭게 생성된 현재 그룹에 저장
                    current_group.append((label, text))   
                
                # 현재 목차가 사실 목차 보다 하위 목차(blv =< lv, no_change)
                else:
                    # 그룹아이디 변경 X
                    current_group.append((label, text))
           
            elif label == 'CONTENT':
                current_group.append((label, text))

        if current_group:
            result.extend([(group_id, l, t) for l, t in current_group])

        return result
    
    # 2024.6.22. modified
    def split_content_sent(self, data):
        new = []
        split_sent = self.split_sent
        bracket = self._bracket
        
        for line in data:
            # Bracket function to split the line
            bracket_split = bracket(line)
            for sub_line in bracket_split:
                if len(split_sent(sub_line)) > 1:
                    for sent in split_sent(sub_line):
                        new.append(sent)
                else:
                    new.append(sub_line)
                    
        return new
