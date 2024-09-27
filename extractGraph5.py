import pandas as pd
import numpy as np

from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
import warnings
warnings.filterwarnings("ignore")
def text2dict(model,TEXT,prompt_dict):
    action_schemas = [
        ResponseSchema(name='ActionReason', description=prompt_dict['ps_action_schemas_ActionReason']),
        ResponseSchema(name='ActionObject', description=prompt_dict['ps_action_schemas_ActionObject']),
        ResponseSchema(name='Action',description=prompt_dict['ps_action_schemas_Action']),
        ResponseSchema(name='ActionResult',description=prompt_dict['ps_action_schemas_ActionResult']),
        ResponseSchema(name='score', description=prompt_dict['ps_action_schemas_score'])
    ]
    event_schemas = [
        ResponseSchema(name='event',description=prompt_dict['ps_event_schemas_event']),
        ResponseSchema(name='subtask', description=prompt_dict['ps_event_schemas_subtask']),
        ResponseSchema(name='score', description=prompt_dict['ps_event_schemas_score'])
    ]

    condition_schemas = [
        ResponseSchema(name='condition',description=prompt_dict['ps_condition_schemas_condition']),
        ResponseSchema(name='causedBy', description=prompt_dict['ps_condition_schemas_causedBy']),
        ResponseSchema(name='score', description=prompt_dict['ps_condition_schemas_score'])
    ]

    feedback_schemas = [
        ResponseSchema(name='ActionResult', description=prompt_dict['ps_feedback_schemas_ActionResult']),
        ResponseSchema(name='feedback',description=prompt_dict['ps_feedback_schemas_feedback']),
        ResponseSchema(name='score', description=prompt_dict['ps_feedback_schemas_score'])
    ]

    action_parser = StructuredOutputParser.from_response_schemas(action_schemas)
    action_instructions = action_parser.get_format_instructions()
    event_parser = StructuredOutputParser.from_response_schemas(event_schemas)
    event_instructions = event_parser.get_format_instructions()
    condition_parser = StructuredOutputParser.from_response_schemas(condition_schemas)
    condition_instructions = condition_parser.get_format_instructions()
    feedback_parser = StructuredOutputParser.from_response_schemas(feedback_schemas)
    feedback_instructions = feedback_parser.get_format_instructions()
    temp=prompt_dict['ps_action_template']
    prompt_action = PromptTemplate(
        template=temp,
        input_variables=["text"],
        partial_variables={"format_instructions": action_instructions}
    )
    temp=prompt_dict['ps_event_template']
    prompt_event = PromptTemplate(
        template=temp,
        input_variables=["text"],
        partial_variables={"format_instructions": event_instructions}
    )
    temp=prompt_dict['ps_condition_template']
    prompt_condition = PromptTemplate(
        template=temp,
        input_variables=["text"],
        partial_variables={"format_instructions": condition_instructions}
    )
    temp=prompt_dict['ps_feedback_template']
    prompt_feedback = PromptTemplate(
        template=temp,
        input_variables=["text"],
        partial_variables={"format_instructions": feedback_instructions}
    )
    input1 = prompt_action.format_prompt(text=TEXT)
    input2 = prompt_condition.format_prompt(text=TEXT)
    input3 = prompt_event.format_prompt(text=TEXT)
    input4 = prompt_feedback.format_prompt(text=TEXT)

    output1 = model(input1.to_string())
    output2 = model(input2.to_string())
    output3 = model(input3.to_string())
    output4 = model(input4.to_string())
    try:
        a1 = action_parser.parse(output1)
    except:
        a1={'score':0}
    try:
        a2 = condition_parser.parse(output2)
    except:
        a2 ={'score':0}
    try:
        a3 = event_parser.parse(output3)
    except:
        a3 = {'score':0}
    try:
        a4 = feedback_parser.parse(output4)
    except:
        a4 = {'score':0}
    return a1,a2,a3,a4,TEXT
def score_action(model_check,TEXT,DICT,prompt_dict):
    score_schemas = [
        ResponseSchema(name='score2', description=prompt_dict['psc_action_score'])
    ]
    score_parser = StructuredOutputParser.from_response_schemas(score_schemas)
    score_instructions = score_parser.get_format_instructions()
    temp=prompt_dict['psc_action_template']
    prompt_score = PromptTemplate(
        template=temp,
        input_variables={"text","dict"},
        partial_variables={"format_instructions":score_instructions}
    )
    input_score= prompt_score.format_prompt(text=TEXT,dict=DICT)
    output_score = model_check(input_score.to_string())
    b1=score_parser.parse(output_score)
    score2=int(b1['score2'])
    return score2
def score_condition(model_check,TEXT,DICT,prompt_dict):
    score_schemas = [
        ResponseSchema(name='score2', description=prompt_dict['psc_condition_score'])
    ]
    score_parser = StructuredOutputParser.from_response_schemas(score_schemas)
    score_instructions = score_parser.get_format_instructions()
    temp=prompt_dict['psc_condition_template']
    prompt_score = PromptTemplate(
        template=temp,
        input_variables={"text","dict"},
        partial_variables={"format_instructions":score_instructions}
    )
    input_score= prompt_score.format_prompt(text=TEXT,dict=DICT)
    output_score = model_check(input_score.to_string())
    b1=score_parser.parse(output_score)
    score2=int(b1['score2'])
    return score2

def score_event(model_check,TEXT,DICT,prompt_dict):
    score_schemas = [
        ResponseSchema(name='score2', description=prompt_dict['psc_event_score'])
    ]
    score_parser = StructuredOutputParser.from_response_schemas(score_schemas)
    score_instructions = score_parser.get_format_instructions()
    temp=prompt_dict['psc_event_template']
    prompt_score = PromptTemplate(
        template=temp,
        input_variables={"text","dict"},
        partial_variables={"format_instructions":score_instructions}
    )
    input_score= prompt_score.format_prompt(text=TEXT,dict=DICT)
    output_score = model_check(input_score.to_string())
    b1=score_parser.parse(output_score)
    score2=int(b1['score2'])
    return score2

def score_feedback(model_check,TEXT,DICT,prompt_dict):
    score_schemas = [
        ResponseSchema(name='score2', description=prompt_dict['psc_feedback_score'])
    ]
    score_parser = StructuredOutputParser.from_response_schemas(score_schemas)
    score_instructions = score_parser.get_format_instructions()
    temp=prompt_dict['psc_feedback_template']
    prompt_score = PromptTemplate(
        template=temp,
        input_variables={"text","dict"},
        partial_variables={"format_instructions":score_instructions}
    )
    input_score= prompt_score.format_prompt(text=TEXT,dict=DICT)
    output_score = model_check(input_score.to_string())
    b1=score_parser.parse(output_score)
    score2=int(b1['score2'])
    return score2

def score_range(a_action, a_condition, a_event, a_feedback, TEXT,model_list,prompt_dict):
    length=len(model_list)
    score_all={}
    text_dict_score_all=[]
    if a_action['score']!=0:
        # print("对action打分")
        score_collect=[int(a_action['score'])]
        for i in range(length):
            try:
                score=score_action(model_list[i],TEXT,a_action,prompt_dict)
                score_collect.append(score)
            except:
                score_collect.append(False)
                pass
        text_dict_score_all.append([TEXT,a_action]+score_collect)
        score_all['action']=score_collect
        if False in score_collect:
            score_collect.remove(False)
        a_action['score']=np.mean(score_collect)
        text_dict_score_all[-1].append(np.mean(score_collect))  # 最后一项是均值
    else:
        a_action['score'] = 0
    if a_condition['score']!=0:
        # print("对condition打分")
        score_collect=[int(a_condition['score'])]
        for i in range(length):
            try:
                score=score_action(model_list[i],TEXT,a_condition,prompt_dict)
                score_collect.append(score)
            except:
                score_collect.append(False)
                pass
        text_dict_score_all.append([TEXT, a_condition] + score_collect)
        score_all['condition'] = score_collect
        if False in score_collect:
            score_collect.remove(False)
        a_condition['score']=np.mean(score_collect)
        text_dict_score_all[-1].append(np.mean(score_collect))  #最后一项是均值
    else:
        a_condition['score'] = 0

    if a_event['score'] != 0:
        # print("对event打分")
        score_collect = [int(a_event['score'])]
        for i in range(length):
            try:
                score = score_action(model_list[i], TEXT, a_event,prompt_dict)
                score_collect.append(score)
            except:
                score_collect.append(False)
                pass
        text_dict_score_all.append([TEXT, a_event] + score_collect)
        score_all['event'] = score_collect
        if False in score_collect:
            score_collect.remove(False)
        a_event['score'] = np.mean(score_collect)  #打分
        text_dict_score_all[-1].append(np.mean(score_collect))  #最后一项是均值

    else:
        a_event['score'] = 0

    if a_feedback['score'] != 0:
        # print("对feedback打分")
        score_collect = [int(a_feedback['score'])]
        for i in range(length):
            try:
                score = score_action(model_list[i], TEXT, a_feedback,prompt_dict)
                score_collect.append(score)
            except:
                score_collect.append(False)
                pass
        text_dict_score_all.append([TEXT, a_feedback] + score_collect)
        score_all['feedback'] = score_collect
        if False in score_collect:
            score_collect.remove(False)
        a_feedback['score'] = np.mean(score_collect)
        text_dict_score_all[-1].append(np.mean(score_collect))  # 最后一项是均值
    else:
        a_feedback['score'] = 0

    collect=[a_action, a_condition, a_event, a_feedback]
    sorted_data = sorted(collect, key=lambda x: x['score'], reverse=True)
    # print('text_dict_score_all=',text_dict_score_all)
    return sorted_data[0],score_all,text_dict_score_all  #取得分最大的那一个,返回全部打分
def tranfer2df(df_action,df_condition, df_event,df_feedback,dict, TEXT):
    dict['text'] = TEXT
    first_key = next(iter(dict.keys()))
    df = pd.DataFrame(dict, index=[0])
    if first_key == 'ActionReason':
        df_action = pd.concat([df_action, df], ignore_index=True)
    if first_key == 'condition':
        df_condition = pd.concat([df_condition, df], ignore_index=True)
    if first_key == 'event':
        df_event = pd.concat([df_event, df], ignore_index=True)
    if first_key == 'ActionResult':
        df_feedback = pd.concat([df_feedback, df], ignore_index=True)
    return df_action, df_condition, df_event, df_feedback
def excel_to_dict(excel_file):
    df = pd.read_excel(excel_file)
    data_dict = dict(zip(df['prompt_template_name'], df['prompt_statement']))
    return data_dict

if __name__=="__main__":
    path_modelSetting = r"E:\BaiduSyncdisk\AIAgentForPilots\AIcode\source\model_list.xlsx"
    model_list = []
    models_setting = pd.read_excel(path_modelSetting).values
    model_gpt4 = OpenAI(openai_api_key=models_setting[0][1], base_url=models_setting[0][2],
                        temperature=models_setting[0][3], top_p=models_setting[0][4])
    for i in range(1, len(models_setting)):
        model_temp = OpenAI(openai_api_key=models_setting[i][1], base_url=models_setting[i][2],
                            temperature=models_setting[i][3], top_p=models_setting[i][4])
        model_list.append(model_temp)
###读取提示词
    # excel_file = r'source/template1_cn.xlsx'
    excel_file = r"E:\BaiduSyncdisk\AIAgentForPilots\AIcode\source\template4_第一版英文提示词模板.xlsx"
    prompt_dict= excel_to_dict(excel_file)
###
    columns = ['ActionReason', 'ActionObject', 'Action', 'ActionResult', 'score', 'text']
    df_action = pd.DataFrame(columns=columns)
    columns = ['condition', 'causedBy', 'score', 'text']
    df_condition = pd.DataFrame(columns=columns)
    columns = ['event', 'subtask', 'score', 'text']
    df_event = pd.DataFrame(columns=columns)
    columns = ['ActionResult', 'feedback', 'score', 'text']
    df_feedback = pd.DataFrame(columns=columns)
###
    dict_action,dict_condition,dict_event,dict_feedback=[],[],[],[]
    text_dict_scoreSave=[]
    TEXT = ["冷舱程序，先执行启动引擎。在推力杆的下方，将引擎模式选择成lgn/Start",'点击按钮，显示器显示为黄色']
    for data in TEXT:
        # 第一步，转换为词典
        a_action, a_condition, a_event, a_feedback, TEXT=text2dict(model_gpt4,data,prompt_dict)
        print("-"*10)
        print('知识提取：',data)
        print(a_action)
        print(a_condition)
        print(a_event)
        print(a_feedback)
        print("-" * 10)
        #第二步，其他模型对转换过程打分，取平均值最高的那个
        dict_after_score,score_all,text_dict_score_all=score_range(a_action, a_condition, a_event, a_feedback, data, model_list,prompt_dict)
        text_dict_scoreSave.extend(text_dict_score_all)
        print("打分结果")
        for i,j in score_all.items():
            print(i,j)
        print(dict_after_score)
        print("-" * 10)
        #第三步，保存
        df_action, df_condition, df_event, df_feedback=tranfer2df(df_action, df_condition, df_event, df_feedback, dict_after_score,data)
    print("*"*10+'显示全部'+"*"*10)
    print(df_action)
    print(df_event)
    print(df_feedback)
    print(df_condition)
    temp = [{'crew': 'pilot', 'note': None}, {'crew': 'pilot2', 'note': None}, ]  # 列表对应的是第一维，即行，字典为同一行不同列元素
    df_pilot = pd.DataFrame(temp)  # 第 1 行 3 列没有元素，自动添加 NaN (Not a Number)
    filename='data_output/test1.xlsx'
    with pd.ExcelWriter(filename) as writer:
        df_pilot.to_excel(writer, 'crewman')
        df_action.to_excel(writer, 'action')
        df_event.to_excel(writer, 'event')
        df_feedback.to_excel(writer, 'feedback')
        df_condition.to_excel(writer, 'condition')

    df_scoreResult=pd.DataFrame(text_dict_scoreSave)
    df_scoreResult.to_excel('data_output/scoreall1.xlsx')

    #第四步，相似词合并说法
# 示例：将Excel文件转换为字典







