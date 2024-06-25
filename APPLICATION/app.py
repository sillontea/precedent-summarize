from pydantic_core.core_schema import NoneSchema
from langchain_openai import OpenAI
import streamlit as st
import os, json, dotenv, tempfile 
from pathlib import Path
import pandas as pd

import sys
sys.path.append('SUMMARIZER')
sys.path.append('EXTRACTOR')
sys.path.append('HIGHLIGHTER')

from extractor import *
from similarity import *
from fact_extractor import *
from utils import *
from summary import *

# API key input
# load the .env file and invoke the secret API key from the file
dotenv.load_dotenv("API.env")
OpenAI.api_key = os.getenv("OPEN_API_KEY")


# Set page config and title
st.set_page_config(page_title="판결문 요약 서비스", page_icon=":book:", layout="wide")
st.title("판결문 요약 서비스")

# Style
# Title style
css = """
        <style>
        .title-text {
            font-size: 24px;
            font-weight: bold;
            text-align: left;
            margin-bottom: 20px;
        }
        </style>
        """

# data
prec_df = pd.read_excel("DATA/precedents.xlsx")
caseid_df = pd.read_excel("DATA/caseNum-id_list.xlsx") # '판례일련번호', '사건번호'
caseid_df = caseid_df[caseid_df['판례일련번호'].isin(prec_df['판례일련번호'].tolist())]


##################

# api 
user = None

# 세션 상태 초기화
if 'caseid' not in st.session_state:
    st.session_state.caseid = "판례일련번호"
    st.session_state.casenum = "사건번호 및"
    st.session_state.precedent = ''
    st.session_state.summary = ''
    st.session_state.attentions = ''
    st.session_state.summ = ''
    st.session_state.hold = ''
    
# Streamlit app(main function)
def main():
    # CSS 스타일 적용
    st.markdown(css, unsafe_allow_html=True)

    # LOGO
    # link -> github
    # st.logo(LOGO_URL_LARGE, link="https://streamlit.io/gallery", icon_image=LOGO_URL_SMALL)
    
    # '판례일련번호', '판시사항', '판결요지', '전문', '참조조문', '참조판례'
    # precedents = prec_df.to_dict(orient='list')
    

    
    # 사이드 바
    with st.sidebar:
        with st.popover("API USER(OC)"):
            st.markdown("API USER(OC) 정보를 입력하세요. 🐣")
            user = st.text_input("USER INFO")
            
        if not user:
            st.warning("정상 동작을 위해 반드시 필요합니다.")

        if st.session_state['precedent'] is None:
            st.info("요약할 판결문을 선택해 주세요.", icon="✍️")

        # 검색 창
        select_table = dict(caseid_df[['사건번호', '판례일련번호']].values)
        selected = st.selectbox("sample_file", sorted(select_table.keys()), index=0) # 사건번호 선택
        
        if st.button("select") and selected:
            case_id = select_table[selected]
            st.session_state.caseid = case_id# 판례일련번호 세션 저장

            st.session_state.casenum = caseid_df[caseid_df['판례일련번호']==case_id].iloc[0]['사건번호']
            st.session_state.precedent = prec_df[prec_df['판례일련번호']==case_id].iloc[0]['전문']
            st.session_state.summary = prec_df[prec_df['판례일련번호']==case_id].iloc[0]['판결요지']
            st.session_state.holding = prec_df[prec_df['판례일련번호']==case_id].iloc[0]['판시사항']


        st.markdown("### 참조판례 및 인용 조문 정보")
        law_extractor = LawExtractor(user)
        
        precedent = st.session_state.precedent
                
        with st.expander("참조 조문", expanded=True):    
            urls, law_names = law_extractor.extract_law_info(precedent)
            law_infos = law_extractor.text_post_processing(urls, law_names)
            
            if urls :
                for i, (law_name, (url, text)) in enumerate(law_infos.items()):
                    _url = ''.join(url.split('type=XML&'))+"&type=HTML"
                    with st.popover(f"{law_name}"):
                        st.markdown(f"[{law_name}]({_url})")
                        st.markdown(f"<span style='font-size: 13px;'>{text}</span>", unsafe_allow_html=True)
            else:
                st.write("참조조문이 존재하지 않습니다.\n")

                
        with st.expander("참조 판례", expanded=True):
            prec_extractor = PrecExtractor()
            ref_prec = prec_extractor.extract(precedent)
            if ref_prec:
                for rp in ref_prec:
                    st.write(rp)
            else:
                st.write("참조판례가 존재하지 않습니다.\n")

            
    case_num, case_id = st.session_state.casenum, st.session_state.caseid    
    
    # 사건번호 및 부가 정보 제공
    st.subheader(f'{case_num} ({case_id})🧑🏻‍⚖️', divider='gray')
    
    # 중요 사실 정보 제공
    with st.expander("사실 정보", expanded=False):
        precedent = st.session_state.precedent
        
        if precedent != '':
            fact_finder = FactExtractor(precedent)
            st.session_state['factfinder'] = fact_finder
        else:
            fact_finder = None
        
        if 'factfinder' in st.session_state:
            if len(fact_finder.facts)==0:
                st.write("추출된 사실이 없습니다.")
            
            elif fact_finder.impfacts:
                st.markdown("#### Important Facts:")
                for fact in fact_finder.impfacts:
                    st.markdown(f"- {fact}")
            else:
                st.markdown("#### Facts:")
                for fact in fact_finder.facts:
                    st.markdown(f"- {fact}")
        else:
            st.write("판결문이 선택되지 않았습니다.")
            
           
    
    tab1, tab2, tab3 = st.tabs(["요약", "판결요지", "목차"])

    with tab1:
        # Layout : 세로 두 열로 분리
        col1, col2 = st.columns([1, 2])

        # 왼쪽 열에 요약 텍스트 표시
        with col1:
            precedent = st.session_state.precedent
            summary = st.session_state.summary
            attentions = st.session_state.attentions
            
            # 화면 표시
            st.markdown("<h3>요약 텍스트</h3>", unsafe_allow_html=True)

            if precedent != '':
                fact_finder = st.session_state.factfinder

                summarizer = Summarizer() 
                precedent = '\n'.join(fact_finder.grouped_data[:, 2])
                summary, attentions = summarizer.summarize(precedent) # attentions, input_tokens pair
                
                input_tokens = summarizer.preprocess_data(precedent)['input_ids'][0]
                input_tokens = list(map(lambda x : summarizer.tokenizer.decode(x, skip_special_tokens=True), input_tokens.cpu()))

                # summary 후처리
                summary = summary.replace('[1]', '').replace('[2]', '').split('\n')
                summary = '\n'.join(list(map(str.strip, summary)))
                
                # 요약 후 세션에 저장
                st.session_state.summary = summary
                st.session_state.attentions = (attentions, input_tokens)

            if st.session_state.summary != '':
                txt = st.text_area(
                "Summarized: ", st.session_state.summary.split('\n')[0]
                , height=300)
            else:    
                txt = st.text_area("Summarized: ", )
            
            # 글자 수 확인
            text_len = (len(txt)-len("Summarized: ")) if (len(txt)-len("Summarized: "))>0 else 0
            st.write(f"We wrote {text_len} characters.")

            
        # 오른쪽 열에 전체 문서 텍스트 표시
        with col2:
            st.markdown("<h3>전체 판결문</h3>", unsafe_allow_html=True)
            container = st.container(border=True, height=1000)
            
            # 체크 박스
            origin = st.checkbox("원본 판결문")
            precedent = st.session_state.precedent
            
            if origin and precedent != '':
                with container:
                    for line in precedent.split('\n'):
                        st.write(line)
            else:
                if precedent != '':
                     with container:  
                        summarizer = Summarizer()
                        precedent = st.session_state['prec']
                        # 본문 하이라이트
                        attentions, input_tokens = st.session_state['attentions']
                        html_text = summarizer.highlight_text(input_tokens, attentions)
                        html_text = html_text[470:]+precedent[len(input_tokens)-11-14:] # prefix 제외
                        html_text += "</div>"

                        # 판결문 원본 데이터 조회
                        for line in html_text.split('\n'):
                            st.markdown(line, unsafe_allow_html=True)


    with tab2:        
        case_id = st.session_state.caseid
        precedent = st.session_state.precedent
        
        # 원문 판결요지 및 판시사항
        hold = st.session_state.hold
        summ = st.session_state.summ
        
        col3, col4 = st.columns([1, 2]) 
        
        if precedent != '': 
            fact_finder = st.session_state.factfinder
            prec = prec_df[prec_df['판례일련번호']==case_id]
            hold = list(np.array(process_sample(prec, "판시사항", comma=True))[:,4])
            summ = list(np.array(process_sample(prec, "판결요지", comma=True))[:,4])
        else:
            hold = ""
            summ = ""
            
        with col3:
            st.markdown("<h3>요약 텍스트</h3>", unsafe_allow_html=True)
            if summ:
                txt = st.text_area("Summarized: ", summ, hold, height=300)
            else:
                txt = st.text_area("Summarized: ", key="blank")
            
            # 글자 수 확인
            text_len = (len(txt)-len("Summarized: ")) if (len(txt)-len("Summarized: "))>0 else 0
            st.write(f"We wrote {text_len} characters.")
            
            _k = st.select_slider("Select Top-K value", options=[i for i in range(1, 11)])
            _type = st.selectbox("Mark type", options=["all", "summ", "hold"])
            
            st.session_state['k'] = _k
            st.session_state['type'] = _type
            
        with col4:
            st.markdown("<h3>전체 판결문</h3>", unsafe_allow_html=True)
            
            if 'k' in st.session_state:
                _k, _type = st.session_state.k, st.session_state.type
            
            if precedent != '':    
                fact_finder = st.session_state.factfinder
                html = show_prec(prec, summ, hold, _type=_type, top_k=_k)
                container = st.container(border=True, height=1000)

                st.session_state['headings'] = [i for i in np.where(fact_finder.grouped_data[: , 1]=='HEAD')][0]

                for i, (line1, line2) in enumerate(zip(fact_finder.grouped_data[:, 2], html.split('<br><br>'))):
                    if i in st.session_state['headings']:
                        container.markdown(f'<span style="background-color: #f0f0f0; padding: 10px;">**{line1}**</span>', unsafe_allow_html=True)
                    else:
                        container.markdown(line2, unsafe_allow_html=True)
                   
                        
    with tab3:
        precedent = st.session_state.precedent
       
        col5, col6 = st.columns([1, 2]) 
        with col5:
            st.markdown("<h3>목차</h3>", unsafe_allow_html=True)
            if precedent != '':
                fact_finder = st.session_state.factfinder

                for head in fact_finder.heads:
                    st.markdown(str(head[2]), unsafe_allow_html=True)
                    
                st.session_state['headings'] = [i for i in np.where(fact_finder.grouped_data[: , 1]=='HEAD')][0]

        with col6:       
            st.markdown("<h3>전체 판결문</h3>", unsafe_allow_html=True)
            container = st.container(border=True, height=1000)
            if precedent != '':
                fact_finder = st.session_state.factfinder
                for i, line in enumerate(fact_finder.grouped_data[:, 2]):
                    if i in st.session_state['headings']:
                        container.markdown(f'<span style="background-color: #f0f0f0; padding: 10px;">**{line}**</span>', unsafe_allow_html=True)
                    else:
                        container.write(line)


            
if __name__ == "__main__":
    main()