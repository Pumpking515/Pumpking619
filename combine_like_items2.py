import pandas as pd
import numpy as np
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
import spacy
import warnings
warnings.filterwarnings("ignore")

def semantic_similarity(sentence1, sentence2):
    nlp = spacy.load("en_core_web_sm")
    doc1 = nlp(sentence1)
    doc2 = nlp(sentence2)
    similarity_score = doc1.similarity(doc2)
    return similarity_score
def generate_newSentense(model,s1,s2,promptStatement_replaceTwoSentenceWithOne):
    schemas1 = [
        ResponseSchema(name='new_sentence', description=promptStatement_replaceTwoSentenceWithOne)
    ]
    parser1 = StructuredOutputParser.from_response_schemas(schemas1)
    instructions1 = parser1.get_format_instructions()
    prompt1 = PromptTemplate(
        template=("You convert two similar short texts{text1}{text2}into one short text,output format{format_instructions};"),
        input_variables={"text1","text2","dict"},
        partial_variables={"format_instructions":instructions1}
    )
    input1= prompt1.format_prompt(text1=s1,text2=s2)
    output = model(input1.to_string())
    temp=parser1.parse(output)
    print("first sentense:",s1)
    print("second sentense:",s2)
    print("new sentense:",temp['new_sentence'])
    return temp['new_sentence']

def judge_and_generate(list1,list2,score_usedtoJudegeSimilarity,model,promptStatement_replaceTwoSentenceWithOne):
    if len(list1)>=1 and len(list2)>=1:
        i_collect=[]
        j_collect=[]
        for i in  range(len(list1)):
            for j in range(len(list2)):
                if i in i_collect:
                    continue
                if j in j_collect:
                    continue
                score=semantic_similarity(list1[i],list2[j])
                print("similarity:",score,'——',list1[i],'——',list2[j])
                if score>score_usedtoJudegeSimilarity:
                    try:
                        new_sentense=generate_newSentense(model,list1[i],list2[j],promptStatement_replaceTwoSentenceWithOne)
                        list1[i]=new_sentense
                        list2[j]=new_sentense
                        i_collect.append(i)
                        j_collect.append(j)
                    except:
                        continue
    return list1,list2

def combine(source_file,save_file,model,score_usedtoJudegeSimilarity,promptStatement_replaceTwoSentenceWithOne):
    # 前奏：读取文件
    data = pd.read_excel(source_file, sheet_name=None)
    df_action = data['action']
    df_feedback = data['feedback']
    df_event = data['event']
    df_crewman = data['crewman']
    df_condition = data['condition']
    # 第一步，针对event,event中的subtask与action中actionReason比较，subtask与condition中的condition比较
    subtask = df_event['subtask'].values
    actionReasion = df_action['ActionReason'].values
    subtask, actionReasion = judge_and_generate(subtask, actionReasion, score_usedtoJudegeSimilarity,model,promptStatement_replaceTwoSentenceWithOne)
    condition = df_condition['condition'].values
    subtask, condition = judge_and_generate(subtask, condition, score_usedtoJudegeSimilarity,model,promptStatement_replaceTwoSentenceWithOne)
    # 更新
    df_event['subtask'] = subtask
    df_action['ActionReason'] = actionReasion
    df_condition['condition'] = condition
    # 第二步，针对action，action中的actionreason与actionresult比较，将action中actionresult与condition中的causedby比较（目的是推进非动作的因果事件）
    actionReasion = df_action['ActionReason'].values
    actionResult = df_action['ActionResult'].values
    actionReasion, actionResult = judge_and_generate(actionReasion, actionResult, score_usedtoJudegeSimilarity,model,promptStatement_replaceTwoSentenceWithOne)
    causedBy = df_condition['causedBy'].values
    actionResult, causedBy = judge_and_generate(actionResult, causedBy, score_usedtoJudegeSimilarity,model,promptStatement_replaceTwoSentenceWithOne)
    # 更新
    df_action['ActionReason'] = actionReasion
    df_action['ActionResult'] = actionResult
    df_condition['causedBy'] = causedBy
    # 第三步，将action中的actionresult与feedback中的actionresult比较
    actionResult = df_action['ActionResult'].values
    actionResult_f = df_feedback['ActionResult'].values
    actionResult, actionResult_f = judge_and_generate(actionResult, actionResult_f, score_usedtoJudegeSimilarity,model,promptStatement_replaceTwoSentenceWithOne)
    # 更新
    df_action['ActionResult'] = actionResult
    df_feedback['ActionResult'] = actionResult_f
    # 尾声：另存为新文件
    with pd.ExcelWriter(save_file) as writer:
        df_crewman.to_excel(writer, 'crewman')
        df_action.to_excel(writer, 'action')
        df_event.to_excel(writer, 'event')
        df_feedback.to_excel(writer, 'feedback')
        df_condition.to_excel(writer, 'condition')

if __name__=="__main__":
    key_gpt4='sk-zk2c07669ea52e38a3f6f43b0ce2f18284f65fbee33679c8'
    url = "https://flag.smarttrot.com/v1/"
    model_gpt4=OpenAI(openai_api_key=key_gpt4,base_url=url,temperature=0.9)
    source_file = r"E:\BaiduSyncdisk\AIAgentForPilots\AIcode\data_output\test8.xlsx"
    save_file = 'data_output/test9.xlsx'

    promptStatement_replaceTwoSentenceWithOne = "Generate a very short sentence to replace two short sentences with the same meaning"
    score_usedtoJudegeSimilarity=0.6
    combine(source_file, save_file, model_gpt4, score_usedtoJudegeSimilarity,promptStatement_replaceTwoSentenceWithOne)









