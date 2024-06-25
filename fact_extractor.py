from utils import SectionSplitter, SentenceSplitter
from classifier import Classifier
import numpy as np
import re

class FactExtractor:
    # extractor.fact_candi, extractor.impfact_candi 에 사실, 중요사실 후보 문장들(해당 목차에 포함된)이 저장
    # extractor.facts, extractor.impfacts에는 각각 위의 후보 문장들 중에서 분류기가 "사실"로 분류한 애들만 저장
    def __init__(self, precedent):
        self.section_splitter = SectionSplitter(precedent)
        self.sentence_splitter = SentenceSplitter()
        self.classifier = Classifier()

        self.grouped_data = np.array(self.section_splitter.grouped_data)
        self.heads = self.get_heads()

        self.extra_group_id = self.get_extra_group_id()
        self.fact_group_id = self.get_fact_group_id()
        self.impfact_group_id = self.get_impfact_group_id()

        self.fact_candi = self.extract_sentences(self.fact_group_id)
        self.impfact_candi = self.extract_sentences(self.impfact_group_id)

        self.facts = self.extract_facts(self.fact_candi)
        self.impfacts = self.extract_facts(self.impfact_candi)

    def get_heads(self):
        head_idx = np.where(self.grouped_data == 'HEAD')[0]
        return self.grouped_data[head_idx]

    def get_extra_group_id(self):
        extra_head = [self.is_extra_head(head) for head in self.heads[:, 2]]
        return [head[0] for head in self.heads[extra_head] if head[0] != '0']

    def get_fact_group_id(self):
        fact_head = [self.is_fact_head(head) for head in self.heads[:, 2]]
        fact_group_id = {head[0] for head in self.heads[fact_head] if head[0] not in self.extra_group_id}
        return sorted(fact_group_id)

    def get_impfact_group_id(self):
        fact_heads = self.heads[np.isin(self.heads[:, 0], self.fact_group_id)]
        impfact_head = [self.is_judge_head(head) for head in fact_heads[:, 2]]
        impfact_group_id = {head[0] for head in fact_heads[impfact_head]}
        return sorted(impfact_group_id)
    
    def extract_sentences(self, group_ids):
        sentences = [list(self.grouped_data[np.isin(self.grouped_data[:, 0], group_id)][:, 1:]) for group_id in group_ids]
        sentences = [np.array(sentence)[np.where(np.array(sentence)[:, 0]!='HEAD')] for sentence in sentences]
        # 별지 및 판사 이름 제거
        sentences = [sent[:, 1] for sent in sentences]
        sentences = [fact for sentence in sentences for fact in sentence if not re.search(r'별\s+지', fact)]
        sentences = sentences[:-1] if re.search("판사", sentences[-1]) else sentences
        return sentences

    def extract_facts(self, extracted_sentences):
        return [sentence for group in extracted_sentences for sentence in group if self.classifier.infer(sentence)[0] == 1]

    def is_fact_head(self, head):
        return self.section_splitter.is_fact_head(head) and "원심" not in head

    def is_extra_head(self, head):
        return self.section_splitter._head_check1(head)

    def is_judge_head(self, head):
        return bool(re.search(r'범죄사실|인정사실|사실의인정|판단(의\s*요지)?$', head))
    
    ## 문장 레이블 용###
    def _extract_sentences_with_id(self, group_ids):
        return [self.grouped_data[np.isin(self.grouped_data[:, 0], group_id)][:] for group_id in group_ids]
    
    def _extract_facts_with_id(self, extracted_sentences):
        return [(x, y, sentence) for group in extracted_sentences for (x, y, sentence) in group if self.classifier.infer(sentence)[0] == 1]

    ################

if __name__ == "__main__":
    import pandas as pd

    path = './'

    file_name = "precedents.xlsx"
    prec = pd.read_excel(path+file_name)
    prec.drop("Unnamed: 0", axis=1, inplace=True)

    file_name = "caseNum-id_list.xlsx"
    casename = pd.read_excel(path+file_name)
    casename.drop("Unnamed: 0", axis=1, inplace=True)
    
    sample = prec.iloc[300]
    case_name = casename[casename['판례일련번호']==sample['판례일련번호']]['사건번호'].values[0]
    
    extractor = FactExtractor(sample['전문'])
    print("Facts:")
    [print(fact) for fact in extractor.facts]
    print("\nImportant Facts:")
    [print(fact) for fact in extractor.impfacts]