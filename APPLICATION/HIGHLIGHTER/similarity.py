import re
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from IPython.display import display, HTML

# for morpheme
from konlpy.tag import Okt # Mecab

# SentenceSplitter, SectionSplitter
from utils import *

# tf-idf
def get_tf_idf_matrix(corpus:list, tfidfv):
    """
    Returns Tf-idf-weighted document-term matrix.

    *** Note: Since tokenize is currently space-based,
    you should be aware that morphological analysis may be required for more accurate usage.

    Term Frequency-Inverse Document Frequency(TF-IDF)
    문서 내에서 단어의 중요성을 평가하고 각 단어의 가중치를 계산하는 데 사용됩니다.
    TF-IDF = TF * IDF

    - Term Frequency (TF): 한 문서 내에서 등장하는 단어들의 빈도
    - Inverse Document Frequency (IDF): 단어의 희소성
      IDF(w)=log(전체 문서 수 / 단어 w가 나타난 문서의 수)

    TfidfVectorizer equivalents to CountVectorizer followed by TfidfTransformer.
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

    """
    return tfidfv.transform(corpus)

# get_tf_idf와 역할이 동일하여 삭제
def get_cosine_sim(matrixA, matrixB):
    """
    주의: TfidfVectorizer에서 정규화(l2)가 이루어지므로 linear_kernel과 cosine_similarity의 결과 값이 같습니다.
    # Row-wise dot product
    """
    cosine_sim = linear_kernel(matrixA, matrixB)
    return cosine_sim

def init_tfidf(corpus):
    tokenizer = Okt().morphs
    tfidfv = TfidfVectorizer(tokenizer=tokenizer, token_pattern=None).fit(corpus)
    return tfidfv


# visualization
def highlight_text(sentences, yellow_indices, red_indices, color1="#FFFACD", color2="#FFB347"):
    # 각 문장을 다시 합칠 때 사용
    full_text = ""

    for i, sentence in enumerate(sentences):
        if i in red_indices:
            # red 인덱스 문장을 하이라이트 처리
            full_text += f'<span style="background-color: {color2};">{sentence}</span><br><br>' 
        elif i in yellow_indices:
            # yellow 인덱스 문장을 하이라이트 처리
            full_text += f'<span style="background-color: {color1};">{sentence}</span><br><br>'
        else:
            full_text += sentence + "<br><br>"
    
    return full_text.strip()


def show_prec(prec, summary, holding, _type='all', thresh=None, top_k=1):
    # 사전에 process_sample 처리
    valid_inputs = {'all', 'summ', 'hold'}
    
    if _type not in valid_inputs:
        raise ValueError(f"Invalid input: {input_value}. Valid inputs are: {', '.join(valid_inputs)}")
    
    # 전문의 경우 content만 비교할 수 있도록 추가 처리 (6.20)+
    ss = SectionSplitter(prec)
    prec_gid, prec_tag, prec_sentences = [np.array(ss.grouped_data)[:, i] for i in range(3)]
    org_content_idx = [ i for i, tag in enumerate(prec_tag) if tag=="CONTENT"] # content 문장의 원래 인덱스
    prec = [prec_sentences[i] for i in org_content_idx] # content 문장만 포함
    
    corpus = prec+holding+summary
    tfidfv = init_tfidf(corpus)
    
    prec_matrix = get_tf_idf_matrix(prec, tfidfv)
    hold_matrix = get_tf_idf_matrix(holding, tfidfv)
    summ_matrix = get_tf_idf_matrix(summary, tfidfv)

    similarity1 = get_cosine_sim(summ_matrix, prec_matrix) # 요지 <-> 판결문
    similarity2 = get_cosine_sim(hold_matrix, prec_matrix) # 판시 <-> 판결문
    # similarity3 = get_cosine_sim(summ_matrix, hold_matrix) # 요지  <-> 판시
    
    
    if thresh is not None:
        indices1 = get_indices_with_threshold(similarity1, threshold) # 판결문 <-> 요지
        indices2 = get_indices_with_threshold(similarity2, threshold) # 판결문 <-> 판시
        # indices3 = get_indices_with_threshold(similarity3, threshold) # 요지  <-> 판시
    else:
        indices1 = get_indices_with_topk(similarity1, top_k) # 판결문 <-> 요지
        indices2 = get_indices_with_topk(similarity2, top_k) # 판결문 <-> 판시
        # indices3 = get_indices_with_topk(similarity3, top_k) # 요지  <-> 판시
    
    # 판결문 전문
    sentences = prec_sentences # content만 포함된 prec이 아닌 전문 prec_sentences를 사용
    highlight1, highlight2 = [], []
    # 하이라이팅할 문장 인덱스 리스트
    def extract_highlight(indices):
        try:
            # 판결문 본문 content 비교에 따른 수정 (6.20)+
            return set([org_content_idx[i] for i in np.array(indices)[:, 1]])
        except:
            return [0]

    highlight1, highlight2 = [], []

    if _type in {'all', 'summ'}:
        highlight1 = extract_highlight(indices1)

    if _type in {'all', 'hold'}:
        highlight2 = extract_highlight(indices2)


    # 하이라이팅된 텍스트 생성
    highlighted_text = highlight_text(sentences, highlight1, highlight2)

    # 결과 HTML 출력
    html = f"<html><body>{highlighted_text}</body></html>"
    display(HTML(html))
    return html
    

# print the similar sentences on the screen
def show_extract_sent(basis_data, compare_data, top_k=1, thresh=None):
    """
    thresh 넘는 문장들은 문장 순서대로 출력
    top_k 문장의 경우 유사도 높은 순으로 출력
    두 개 동시에 입력되었을 경우 thresh 우선으로 반환
    default : top_k 1

    """
    basis_matrix, compare_matrix, similarity_matrix = get_matrices(basis_data, compare_data)
    
    if thresh is not None:
        indices = get_indices_with_threshold(similarity_matrix, thresh)
    else:
        indices = get_indices_with_topk(similarity_matrix, top_k)
    
    bi = -1
    bsent, csent = '', ''
    
    for i, j in indices:
        
        bsent = basis_data[i]
        csent = compare_data[j]
        sim_score = similarity_matrix[i][j]

        if (bi != i) or (bi == -1):
            print("====="*20)
            print(f"<<{i+1}번째 기준 문장>>")
            display(f"{bsent}")
            bi = i
            cnt = 1

        print("-----"*15)
        display(f"[{cnt}]{j+1}번째 문장(유사도 {sim_score:.3f})|| {csent}")
        cnt += 1

#####

# whole precess for getting indices for extract similar sentences
def get_matrices(basis_data, compare_data):
    corpus = basis_data+compare_data
    tfidfv = init_tfidf(corpus)

    basis_matrix = get_tf_idf_matrix(basis_data, tfidfv)
    compare_matrix = get_tf_idf_matrix(compare_data, tfidfv)
    similarity_matrix = get_cosine_sim(basis_matrix, compare_matrix)
    
    return basis_matrix, compare_matrix, similarity_matrix
    
    
# search the similar sentences with threshold
def get_indices_with_threshold(similarity_matrix, thresh=0.8):
    """
    Returns:
        numpy array: elements has tuple data pairs of basis sentence's and compare sentence's index
    """
    indices = []
    # (basis sentence index, similar sentence index)
    candidates = list(zip(np.where(similarity_matrix>thresh)[0], np.where(similarity_matrix>thresh)[1]))
    
    for _, similar_sentenc_index in candidates:
        indices.append((_, similar_sentenc_index))
    return np.array(indices)


# search the top_k similar sentences with similarity score
def get_indices_with_topk(similarity_matrix, top_k=1):
    """
    Args:
        similarity_matrix(np.arrray) - cosine similarity matrix
    Returns:
        numpy array: elements has tuple data pairs of basis sentence's and compare sentence's index
    """
    results = []
    
    # 유사도 내림차순 정렬한 인덱스
    sorted_indices = np.argsort(-similarity_matrix)
    
    # 기준 문장과 가장 유사한 top_k개의 문장 인덱스
    top_indices = list(map(lambda x: x[:top_k], sorted_indices))
    
    return [(i, j) for i, _ in enumerate(top_indices) for j in _ ]
    
    
# 판시, 요지, 판결문 문장 단위 및 단락 단위로 분리
def process_sample(sample, col_name, comma=True):
    """
    Preprocessing the summary, holding and precedent before to compare similarities between them.
    Args:
        sample(DataFrame): target precedent dataset formed as pandas DataFrame
        col_name: a target column in DataFrame(sample), which contains str type data.
    Returns:
        list: A list of processed sentences.
    """
    split_sent = SentenceSplitter().split_sent
    
    # [n] 넘버링 제거
    pattern = r"\[[\d]+\]" # [1], [2]와 같은 패턴
    items = [itm.strip() for itm in re.split(pattern, sample[col_name]) if itm != '']

    new_items = []

    for item in items:
        tmp = []
        # 원래 데이터에서 \n(줄바꿈, api에서 </br>) 기준으로 분리
        for i, line in enumerate(item.split('\n')):
            if col_name == "판시사항":
                # 판시사항의 경우 "/" 기준으로 분리
                tmp.extend([itm.strip() for itm in line.split('/') if itm != ''])

            else:
                # 판결요지의 경우 문장을 분리
                # 6.20 판결요지 comma 단위 분리 조건 추가
                if comma:
                    tmp.extend([(i, _.strip()) for itm in split_sent(line)for _ in itm.split(',')])
                else:       
                    tmp.extend([(i, itm.strip()) for itm in split_sent(line) if itm != ''])
        new_items.append(tmp)

    # 새로운 데이터프레임 생성
    result = []
    for i, _item in enumerate(new_items):
        try:
            for j, (k, sentence) in enumerate(_item):
                result.append([sample['판례일련번호'], f"section {i}", k, f"sentence {j}", sentence])

        except:
            for j, sentence in enumerate(_item):
                result.append([sample['판례일련번호'], f"section {i}", j, f"sentence {j}", sentence])
            
    return result
